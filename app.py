import os
import streamlit as st
from processing.document import extract_text, split_documents
from processing.embeddings import EmbeddingGenerator
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import tempfile
import uuid
from typing import List, Dict

# Initialize Google Gemini API
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing Gemini API key in Streamlit secrets")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def init_session() -> None:
    if "processed" not in st.session_state:
        st.session_state.update({
            "user_id": f"user_{uuid.uuid4().hex[:8]}",
            "embedding_generator": EmbeddingGenerator(
                index_name="pagepal-shared",
                api_key=st.secrets.get("PINECONE_API_KEY")
            ),
            "processed": False,
            "uploaded_files": [],
            "messages": []
        })

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
    relevant_matches = [m for m in matches if (m.get('score', 0) >= MIN_SCORE)]
    
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
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        
        if context == "no_relevant_context":
            response = model.generate_content(
                f"Answer concisely: {query}",
                generation_config=GenerationConfig(temperature=0.5)
            )
            return f"Based on general knowledge: {response.text.strip()}"
            
        prompt = build_prompt(query, context)
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(temperature=0.3)
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.title("📚 PagePal: Document Assistant")
    st.markdown("Upload documents & ask questions")
    
    # File uploader (always visible)
    uploaded_files = st.file_uploader(
        "Upload PDF/DOCX/TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Max file size: 10MB"
    )
    
    # Document processing
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
                        st.success(f"Processed {len(all_texts)} sections!")
                        st.experimental_rerun()
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")

    # Chat interface (always visible after processing)
    if st.session_state.processed:
        st.subheader("Chat")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Clear data button
        if st.button("❌ Clear All Data"):
            st.session_state.embedding_generator.delete_all()
            st.session_state.processed = False
            st.session_state.messages = []
            pass

        # Chat input
        if prompt := st.chat_input("Ask about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                try:
                    results = st.session_state.embedding_generator.query_embeddings(
                        query=prompt,
                        top_k=8,
                        filter_dict={"user_id": st.session_state.user_id}
                    )
                    doc_context = build_context(results.get("matches", []))
                    
                    with st.spinner("Analyzing documents..."):
                        response = generate_response(prompt, doc_context)
                        st.markdown(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
