# app.py

import torch
torch.set_default_device('cpu')

import os
import streamlit as st
from processing.document import extract_text, split_documents
from processing.embeddings import EmbeddingGenerator
from google import genai
from google.genai import types, GenerationConfig
import tempfile
import uuid
from typing import List, Dict


# Initialize Google Gemini API
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing Gemini API key in Streamlit secrets")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def init_session() -> None:
    session_defaults = {
        "user_id": f"user_{uuid.uuid4().hex[:8]}",
        "embedding_generator": EmbeddingGenerator(
            index_name="pagepal-shared",
            api_key=st.secrets.get("PINECONE_API_KEY")
        ),
        "chat_history": [],
        "processed": False,
        "uploaded_files": []
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session()

def build_prompt(query: str, context: str) -> str:
    return f"""You are a document expert assistant. Follow these rules:
1. Answer using the document context when possible
2. For general knowledge questions, say "Based on general knowledge:"
3. If unsure, say "This isn't covered in the documents"

Document Context:
{context}

Question: {query}

Answer:"""

def build_context(matches: List[Dict], max_words: int = 2000) -> str:
    if not matches:
        return "no_relevant_context"
    
    MIN_SCORE = 0.15
    relevant_matches = [
        m for m in matches 
        if (m.get('score', 0) >= MIN_SCORE)
    ]
    
    context = []
    total_words = 0
    for match in relevant_matches:
        metadata = match.get('metadata', {})
        text = metadata.get('text', '')
        words = text.split()
        if total_words + len(words) > max_words:
            break
        context.append(f"- {text}")
        total_words += len(words)
    return "\n".join(context) if context else "no_relevant_context"

def generate_response(query: str, context: str) -> str:
    try:
        if context == "no_relevant_context":
            response = genai.generate_content(
                f"Answer concisely: {query}",
                generation_config=GenerationConfig(
                    temperature=0.5
                )
            )
            return f"Based on general knowledge: {response.text.strip()}"
            
        prompt = build_prompt(query, context)
        response = genai.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.3
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.title("📚 PagePal: Document Assistant")
    st.markdown("Upload documents & ask questions")
    
    # File processing
    uploaded_files = st.file_uploader(
        "Upload PDF/DOCX/TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Max file size: 10MB"
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
                        st.session_state.embedding_generator.store_embeddings(
                            all_texts, all_metadata
                        )
                        st.session_state.processed = True
                        st.success(f"Processed {len(all_texts)} sections!")
                        st.session_state.uploaded_files = [f.name for f in uploaded_files]
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")

    # Chat interface
    if st.session_state.processed:
        st.subheader("Chat")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

        if st.button("Clear All Data"):
            st.session_state.embedding_generator.delete_all()
            st.session_state.processed = False
            st.session_state.chat_history = []
            st.rerun()

        if prompt := st.chat_input("Ask about your documents..."):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                try:
                    # Retrieve context
                    results = st.session_state.embedding_generator.query_embeddings(
                        query=prompt,
                        top_k=8,
                        filter_dict={"user_id": st.session_state.user_id}
                    )
                    matches = results.get("matches", [])
                    doc_context = build_context(matches)
                    
                    # Generate and display
                    response = generate_response(prompt, doc_context)
                    st.write(response)
                    
                    # Add to history
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.write(error_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg}
                    )

if __name__ == "__main__":
    main()
