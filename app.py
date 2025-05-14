#app.py
import os
import streamlit as st
from processing.document import extract_text, split_documents
from processing.embeddings import EmbeddingGenerator
from google import genai
from google.genai import types
import tempfile
import uuid
from typing import List, Dict

# Initialize Google Gemini API client
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Google Gemini API key is missing. Please configure secrets.")
    st.stop()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def init_session() -> None:
    if "user_id" not in st.session_state:
        random_id = str(uuid.uuid4())
        st.session_state.user_id = f"user_{random_id[:8]}"
    if "embedding_generator" not in st.session_state:
        index_name = "pagepal-shared"
        st.session_state.embedding_generator = EmbeddingGenerator(
            index_name=index_name,
            api_key=st.secrets["PINECONE_API_KEY"],
            environment=st.secrets["PINECONE_ENVIRONMENT"]
        )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

init_session()

def build_prompt(query: str, context: str, chat_history: str) -> str:
    return f"""You are a helpful document assistant. Use the conversation and context below to answer.

Conversation History:
{chat_history}

Context:
{context}

Question: {query}

Answer in clear, concise English. If unsure, say you don't know. Do not mention the context or documents in your response.
Answer:"""

def build_context(matches: List[Dict], max_words: int = 1500) -> str:
    if not matches:
        return "no_relevant_context"
    MIN_SCORE = 0.25
    relevant_matches = [
        m for m in matches
        if getattr(m, 'score', 0) >= MIN_SCORE or (isinstance(m, dict) and m.get('score', 0) >= MIN_SCORE)
    ]
    if not relevant_matches:
        return "no_relevant_context"
    context = []
    total_words = 0
    for match in relevant_matches:
        if hasattr(match, 'metadata'):
            metadata = match.metadata
        elif isinstance(match, dict):
            metadata = match.get('metadata', {})
        else:
            continue
        text = metadata.get('text', '')
        words = text.split()
        if total_words + len(words) > max_words:
            break
        context.append(f"Document excerpt: {text}")
        total_words += len(words)
    return "\n".join(context) if context else "no_relevant_context"

def answer_general_question(query: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"""Answer the following question in clear, concise English.
            If you don't know, say so. Question: {query}""",
            config=types.GenerateContentConfig(
                response_modalities=["Text"],
                temperature=0.5
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_response(query: str, context: str, chat_history: str) -> str:
    try:
        if context == "no_relevant_context":
            return answer_general_question(query)
        prompt = build_prompt(query, context, chat_history)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["Text"],
                temperature=0.3
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Error: {str(e)}")
        return "I encountered an error processing your request. Please try again."

def get_all_user_chunks(
    embedding_generator: EmbeddingGenerator,
    user_id: str,
    max_chunks: int = 100,
    max_chars: int = 30000
) -> str:
    """Retrieves all document chunks for a user within character limits."""
    index = embedding_generator.index
    dummy_vector = embedding_generator.generate_query_embedding("summary").tolist()
    results = index.query(
        vector=dummy_vector,
        top_k=max_chunks,
        include_metadata=True,
        filter={"user_id": user_id}
    )
    all_texts = []
    total_chars = 0
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        text = meta.get("text", "")
        if total_chars + len(text) > max_chars:
            break
        all_texts.append(text)
        total_chars += len(text)
    return "\n".join(all_texts)

def format_chat_history(history: List[Dict], max_messages: int = 5, max_chars: int = 4000) -> str:
    """Formats the last N chat messages for context, with a char limit."""
    formatted = []
    total_chars = 0
    # Go backwards for most recent, but prepend to keep order
    for msg in reversed(history[-max_messages:]):
        role = "User" if msg["role"] == "user" else "Assistant"
        msg_content = f"{role}: {msg['content']}"
        if total_chars + len(msg_content) > max_chars:
            break
        formatted.insert(0, msg_content)
        total_chars += len(msg_content)
    return "\n".join(formatted)

def main():
    st.title("📚 PagePal - Your Friendly Document Assistant!")
    st.markdown("Upload documents and ask questions. I can also answer general knowledge questions!")

    uploaded_files = st.file_uploader(
        "Upload your study materials (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    # Store uploaded files in session for clearing later
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files

    if uploaded_files and not st.session_state.processed:
        with st.spinner("Processing documents..."):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    all_texts = []
                    all_metadata = []
                    for file in uploaded_files:
                        file_path = os.path.join(temp_dir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        try:
                            docs = extract_text(file_path)
                        except Exception as e:
                            print(f"Error processing file {file.name}: {str(e)}")
                            st.error(f"Failed to extract from {file.name}: {e}")
                            continue
                        chunks = split_documents(docs)
                        for chunk in chunks:
                            all_texts.append(chunk.page_content)
                            all_metadata.append({
                                **chunk.metadata,
                                "user_id": st.session_state.user_id,
                                "document_id": str(uuid.uuid4())
                            })
                    if all_texts:
                        st.session_state.embedding_generator.store_embeddings(
                            all_texts,
                            all_metadata
                        )
                        st.session_state.processed = True
                        st.success(f"Processed {len(all_texts)} document chunks!")
                    else:
                        st.error("No valid text extracted from uploaded files.")
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")

    if st.session_state.processed:
        st.subheader("Chat Interface")
        # Show only last 5 messages
        for msg in st.session_state.chat_history[-5:]:
            st.chat_message(msg["role"]).write(msg["content"])

        if st.button("Clear Database and Restart"):
            # Clear Pinecone index
            st.session_state.embedding_generator.delete_all()
            # Reset session state
            st.session_state.processed = False
            st.session_state.chat_history = []
            st.session_state.uploaded_files = []
            # Rerun to reset file uploader and UI
            st.rerun()

        if prompt := st.chat_input("Ask about your documents or anything else:"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            with st.spinner("Analyzing..."):
                try:
                    chat_history_str = format_chat_history(st.session_state.chat_history)
                    # Get document context
                    if any(s in prompt.lower() for s in ["summarise", "summarize"]):
                        doc_context = get_all_user_chunks(
                            st.session_state.embedding_generator,
                            st.session_state.user_id
                        )
                    else:
                        results = st.session_state.embedding_generator.query_embeddings(
                            query=prompt,
                            top_k=5,
                            filter_dict={"user_id": st.session_state.user_id}
                        )
                        matches = results.get("matches", [])
                        doc_context = build_context(matches)
                    # Combine contexts for LLM
                    full_context = f"{chat_history_str}\n\n{doc_context}"
                    response = generate_response(prompt, doc_context, chat_history_str)
                except Exception as e:
                    response = f"Error processing request: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()
