#processing/document.py
import os
from langchain_community.document_loaders import PyPDFLoader
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def extract_text(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return extract_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_from_docx(file_path)
    elif file_extension == '.txt':
        return extract_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def extract_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    # Returns a list of Document objects, one per page
    return loader.load()

def extract_from_docx(file_path):
    text = docx2txt.process(file_path)
    # Convert to Document object for consistency
    return [Document(page_content=text, metadata={"source": file_path})]

def extract_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    # Convert to Document object for consistency
    return [Document(page_content=text, metadata={"source": file_path})]

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    if isinstance(documents[0], str):
        documents = [Document(page_content=doc) for doc in documents]
    
    return splitter.split_documents(documents)
