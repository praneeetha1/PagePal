# 📚 PagePal: Document Q&A System

Welcome to **PagePal**, a powerful document assistant that lets you have conversations with your uploaded files. Ask questions about your documents and get accurate, contextual answers powered by AI.

---

## 🚀 Features

- **Multi-format Support**: Upload and process PDF, DOCX, and TXT files  
- **Semantic Search**: Find the most relevant content for your questions using vector embeddings  
- **Conversational Interface**: Maintain context through a chat-like experience  
- **Vector Database Integration**: Efficiently store and retrieve document knowledge  
- **AI-Powered Responses**: Generate helpful answers using Google's Gemini AI

---

## ⚙️ How It Works

1. **Document Processing**: Upload your files and PagePal breaks them into manageable chunks  
2. **Embedding Generation**: Text is converted to vector embeddings using SentenceTransformers  
3. **Vector Storage**: Embeddings are stored in Pinecone's vector database  
4. **Semantic Search**: When you ask a question, PagePal finds the most relevant document sections  
5. **Response Generation**: Google's Gemini AI generates responses based on the retrieved context

---

## 🛠 Installation

### 1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/pagepal.git
cd pagepal
