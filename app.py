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
    session_defaults = {
        "user_id": f"user_{str(uuid.uuid4())[:8]}",
        "embedding_generator": EmbeddingGenerator(
            index_name="pagepal-shared",
            api_key=st.secrets["PINECONE_API_KEY"],
            environment=st.secrets["PINECONE_ENVIRONMENT"]
        ),
        "chat_history": [],
        "processed": False,
        "uploaded_files": []
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session()

def build_prompt(query: str, context: str, chat_history: str) -> str:
    return f"""You are a document expert assistant. Follow these rules:
1. Answer using the document context when possible
2. For general knowledge questions, say "Based on general knowledge:"
3. If unsure, say "This isn't covered in the documents"

Recent conversation:
{chat_history}

Document context:
{context}

Question: {query}

Answer:"""

def build_context(matches: List[Dict], max_words: int = 2000) -> str:
    if not matches:
        return "no_relevant_context"
    
    MIN_SCORE = 0.18  # Lowered threshold for better recall
    relevant_matches = [
        m for m in matches 
        if (getattr(m, 'score', 0) >= MIN_SCORE) or 
        (isinstance(m, dict) and m.get('score', 0) >= MIN_SCORE)
    ]
    
    context = []
    total_words = 0
    for match in relevant_matches:
        metadata = match.metadata if hasattr(match, 'metadata') else match.get('metadata', {})
        text = metadata.get('text', '')
        words = text.split()
        if total_words + len(words) > max_words:
            break
        context.append(f"- {text}")
        total_words += len(words)
    return "\n".join(context) if context else "no_relevant_context"

def expand_query(query: str, history: List[Dict]) -> str:
    """Add context from previous 2 user messages"""
    prev_queries = [msg["content"] for msg in history[-3:] if msg["role"] == "user"]
    return " ".join(prev_queries[-2:] + [query])

def generate_response(query: str, context: str, chat_history: str) -> str:
    try:
        if context == "no_relevant_context":
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"Answer like a helpful assistant: {query}",
                config=types.GenerateContentConfig(
                    response_modalities=["Text"],
                    temperature=0.5
                )
            )
            return f"Based on general knowledge: {response.text.strip()}"
            
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
        return f"Error processing request: {str(e)}"

def main():
    st.title("📚 Document Assistant")
    st.markdown("Upload documents & ask questions. I'll use them when possible!")
    
    # File processing
    uploaded_files = st.file_uploader(
        "Upload PDF/DOCX/TXT files",
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
                        docs = extract_text(file_path)
                        chunks = split_documents(docs)
                        for chunk in chunks:
                            all_texts.append(chunk.page_content)
                            all_metadata.append({
                                **chunk.metadata,
                                "user_id": st.session_state.user_id,
                                "document_id": str(uuid.uuid4())
                            })
                    if all_texts:
                        st.session_state.embedding_generator.store_embeddings(all_texts, all_metadata)
                        st.session_state.processed = True
                        st.success(f"Processed {len(all_texts)} document sections!")
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")

    # Chat interface
    if st.session_state.processed:
        st.subheader("Chat")
        for msg in st.session_state.chat_history[-5:]:
            st.chat_message(msg["role"]).write(msg["content"])

        if st.button("Clear All Data"):
            st.session_state.embedding_generator.delete_all()
            st.session_state.processed = False
            st.session_state.chat_history = []
            st.rerun()

        if prompt := st.chat_input("Ask about your documents..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.spinner("Researching..."):
                try:
                    # Enhanced query with conversation context
                    expanded_query = expand_query(prompt, st.session_state.chat_history)
                    
                    # Retrieve document context
                    results = st.session_state.embedding_generator.query_embeddings(
                        query=expanded_query,
                        top_k=8,  # Increased from 5
                        filter_dict={"user_id": st.session_state.user_id}
                    )
                    matches = [m for m in results.get("matches", []) if m.get('score', 0) >= 0.15]
                    doc_context = build_context(matches)
                    
                    # Generate response
                    chat_history = "\n".join(
                        [f"{msg['role']}: {msg['content']}" 
                         for msg in st.session_state.chat_history[-3:]]
                    )
                    response = generate_response(prompt, doc_context, chat_history)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error processing request: {str(e)}")

if __name__ == "__main__":
    main()
