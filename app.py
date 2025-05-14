#app.py
import os
import streamlit as st
from processing.document import extract_text, split_documents
from processing.embeddings import EmbeddingGenerator
from google import genai
from google.genai import types
import tempfile
import uuid

# Initialize Google Gemini API client
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Google Gemini API key is missing. Please configure secrets.")
    st.stop()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def init_session():
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

init_session()

def build_prompt(query: str, context: str) -> str:
    return f"""You are a helpful document assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer in clear, concise English. If unsure, say you don't know. Do not mention the context or documents in your response.
Answer:"""

def build_context(matches, max_words=1500):
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

def generate_response(query: str, context: str) -> str:
    try:
        if context == "no_relevant_context":
            return answer_general_question(query)
        prompt = build_prompt(query, context)
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

def get_all_user_chunks(embedding_generator, user_id, max_chunks=100, max_chars=12000):
    """Get all text chunks for a user, up to max_chunks and max_chars."""
    index = embedding_generator.index
    # Use a generic query vector (e.g., for "summary")
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

def main():
    st.title("📚 PagePal - Your Friendly Document Assistant!")
    st.markdown("Upload documents and ask questions. I can also answer general knowledge questions!")

    uploaded_files = st.file_uploader(
        "Upload your study materials (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

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
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

        if st.button("Clear Documents & Restart"):
            st.session_state.embedding_generator.delete_all()
            st.session_state.processed = False
            st.session_state.chat_history = []
            st.rerun()

        if prompt := st.chat_input("Ask about your documents or anything else:"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            with st.spinner("Analyzing..."):
                try:
                    # Special handling for summarization requests
                    prompt_lower = prompt.strip().lower()
                    if any(
                        s in prompt_lower
                        for s in [
                            "summarise", "summarize",
                            "summarise the document", "summarize the document",
                            "summarise the entire document", "summarize the entire document"
                        ]
                    ):
                        context = get_all_user_chunks(
                            st.session_state.embedding_generator,
                            st.session_state.user_id,
                            max_chunks=100,
                            max_chars=12000
                        )
                        response = generate_response(prompt, context)
                    else:
                        results = st.session_state.embedding_generator.query_embeddings(
                            query=prompt,
                            top_k=5,
                            filter_dict={"user_id": st.session_state.user_id}
                        )
                        matches = results.get("matches", [])
                        context = build_context(matches)
                        response = generate_response(prompt, context)
                except Exception as e:
                    response = f"Error processing request: {str(e)}"
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()
