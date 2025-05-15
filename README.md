# PagePal

Welcome to **PagePal**, your intelligent document assistant designed to help you interact with and explore your uploaded files. With PagePal, you can ask questions about your documents and get precise, contextual answers.

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ihaveexamtomo.streamlit.app/)

Try the app live: [https://ihaveexamtomo.streamlit.app/](https://ihaveexamtomo.streamlit.app/)

## Features

- **Multi-format Support**: Upload and process PDF, DOCX, and TXT files  
- **Semantic Search**: Find the most relevant content for your questions using vector embeddings  
- **Conversational Interface**: Maintain context through a chat-like experience  
- **Vector Database Integration**: Efficiently store and retrieve document knowledge  
- **AI-Powered Responses**: Generate helpful answers using Google's Gemini AI


## ⚙️ How It Works

1. **Document Processing**: Upload your files and PagePal breaks them into manageable chunks  
2. **Embedding Generation**: Text is converted to vector embeddings using SentenceTransformers  
3. **Vector Storage**: Embeddings are stored in Pinecone's vector database  
4. **Semantic Search**: When you ask a question, PagePal finds the most relevant document sections  
5. **Response Generation**: Google's Gemini AI generates responses based on the retrieved context

## 🛠 Installation

### 1. Clone this repository:
```
git clone https://github.com/YOUR_USERNAME/pagepal.git
cd pagepal
```

## 2. Install the required Python packages
```
pip install -r requirements.txt
```

## 3. Setup Environment Variables
 Create a `.env` file with your API keys:
 ```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_index_name
GEMINI_API_KEY=your_gemini_api_key
```

## Usage
1. Run the application:
```
streamlit run app.py
```
2. Upload your documents (PDF, DOCX, TXT)
3. Wait for processing to complete
4. Start asking questions about your documents

## 📋 Requirements

- Python 3.8+
- Pinecone account (for vector database)
- Google Gemini API key

## Notes
PagePal currently works with digital text documents only. It cannot process handwritten notes, scanned images of text, or screenshots without OCR pre-processing
