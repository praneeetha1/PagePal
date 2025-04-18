# app.py
import os
import streamlit as st
from processing.document import extract_text, split_documents
from processing.embeddings import EmbeddingGenerator
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import Any, List

load_dotenv()

# Initialize Google Gemini API client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Google Gemini API key is missing. Please set GEMINI_API_KEY in your .env file.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize session state
if "embedding_generator" not in st.session_state:
    st.session_state.embedding_generator = EmbeddingGenerator()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed" not in st.session_state:
    st.session_state.processed = False
if "summary" not in st.session_state:
    st.session_state.summary = ""

# Helper function to summarize chat history using the LLM
def summarize_history(chat_history):
    try:
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
        gemini_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Summarize the following conversation:\n\n{history_text}",
            config=types.GenerateContentConfig(
                response_modalities=["Text"]
            ),
        )
        return gemini_response.text.strip()
    except Exception as e:
        return f"Error summarizing history: {str(e)}"

def summarize_recent_history(chat_history, max_messages=5):
    recent_history = chat_history[-max_messages:]
    return summarize_history(recent_history)

# Helper function to generate a response using Google Gemini API
def generate_response(prompt, history_summary, context, recent_text):
    try:
        gemini_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"You are a helpful assistant.\n\nSummary of previous conversation:\n{history_summary}\n\nRecent conversation:\n{recent_text}\n\nContext from documents:\n{context}\n\nQuestion: {prompt}\nAnswer:",
            config=types.GenerateContentConfig(response_modalities=["Text"])
        )
        return gemini_response.text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Helper function to generate a fallback response for unrelated questions
def generate_fallback_response(prompt):
    try:
        gemini_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"You are a friendly bot that answers general questions when no relevant information is found in the uploaded documents.\n\nQuestion: {prompt}\nAnswer:",
            config=types.GenerateContentConfig(
                response_modalities=["Text"]
            ),
        )
        return gemini_response.text.strip()
    except Exception as e:
        return f"Error generating fallback response: {str(e)}"

# Main function
def main():
    st.title("📚 Document Q&A System")
    st.markdown("Upload your documents and ask questions about them.")

    # Option to clear Pinecone data
    if st.button("Clear Pinecone Database"):
        try:
            st.session_state.embedding_generator.delete_all()
            st.success("Pinecone database cleared successfully!")
            st.session_state.processed = False
            st.session_state.chat_history = []
            st.session_state.summary = ""
        except Exception as e:
            st.error(f"Error clearing Pinecone database: {e}")

    # File upload section
    uploaded_files = st.file_uploader(
        "Upload your study materials (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files and not st.session_state.processed:
        with st.spinner("Processing your documents..."):
            all_texts = []
            all_metadata = []

            for file in uploaded_files:
                try:
                    os.makedirs("./temp", exist_ok=True)
                    file_path = f"./temp/{file.name}"
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    docs = extract_text(file_path)
                    chunks = split_documents(docs)
                    for chunk in chunks:
                        all_texts.append(chunk.page_content)
                        all_metadata.append(chunk.metadata)
                except Exception as e:
                    st.error(f"Error processing file {file.name}: {e}")
                    continue

            st.session_state.embedding_generator.store_embeddings(all_texts, all_metadata)
            st.session_state.processed = True
            st.success(f"Processed {len(all_texts)} document chunks!")

        # Cleanup temporary files
        import shutil
        shutil.rmtree("./temp", ignore_errors=True)

    # Q&A section
    if st.session_state.processed:
        st.subheader("Chat with Your Documents")

        # Display chat history
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Ask about your documents:"):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            with st.spinner("Generating response..."):
                # Maintain a running summary of the conversation
                N = 5  # Number of recent messages to include
                recent_messages = st.session_state.chat_history[-N:]
                to_summarize = f"Previous summary:\n{st.session_state.summary}\n\nRecent conversation:\n" + \
                    "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_messages])
                try:
                    gemini_response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=f"Update the following summary with the recent conversation below.\n\n{to_summarize}",
                        config=types.GenerateContentConfig(response_modalities=["Text"])
                    )
                    st.session_state.summary = gemini_response.text.strip()
                except Exception as e:
                    st.session_state.summary = f"Error updating summary: {str(e)}"

                history_summary = st.session_state.summary

                # Search for relevant context in the documents
                results = st.session_state.embedding_generator.query_embeddings(
                    query=prompt,
                    top_k=5
                )
                context = "\n".join([
                    f"Document {i+1}: {match['metadata'].get('text','')}"
                    for i, match in enumerate(results.get("matches", []))
                ])

                # Prepare recent conversation text
                recent_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_messages])

                # Generate response
                response = generate_response(prompt, history_summary, context, recent_text)

                # Add assistant message to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()


