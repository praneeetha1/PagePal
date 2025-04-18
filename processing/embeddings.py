# processing/embeddings.py
import os
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

class EmbeddingGenerator:
    ## INITIALIZATION
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
        self.api_key = os.getenv('PINECONE_API_KEY')
        self.environment = os.getenv('PINECONE_ENVIRONMENT')
        self.index_name = os.getenv('PINECONE_INDEX_NAME')

        if not all([self.api_key, self.environment, self.index_name]):
            raise ValueError("Missing Pinecone credentials. Please check your .env file.")

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        # Check if index exists, create if it doesn't
        indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in indexes:
            # Create index - dimension should match your embedding model
            # all-MiniLM-L6-v2 has 384 dimensions
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Created new Pinecone index: {self.index_name}")
        
        # Connect to the index
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to Pinecone index: {self.index_name}")
        
        
    ## EMBEDDING GENERATION
    
    def generate_embeddings(self, texts):
        """Generate embeddings for a list of text chunks"""
        if not texts:
            return np.array([])
            
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts)
            all_embeddings.extend(batch_embeddings)
            print(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return np.array(all_embeddings)
    
    
    def generate_query_embedding(self, query):
        """Generate embedding for a single query"""
        return self.model.encode(query)
    
    
    
    ## STORE EMBEDDINGS

    def store_embeddings(self, texts, metadata_list=None):
        if not texts:
            print("No texts to embed")
            return
            
        embeddings = self.generate_embeddings(texts)
        
        vectors_to_upsert = []
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            metadata = {"text": text}
            
            if metadata_list and i < len(metadata_list):
                metadata.update(metadata_list[i])
                
            vectors_to_upsert.append({
                "id": f"vec_{i}_{hash(text) % 10000}",
                "values": embedding.tolist(),
                "metadata": metadata
            })
            
        # Insert in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i+batch_size]
            self.index.upsert(vectors=batch)
            print(f"Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1}")
        
        print(f"Successfully stored {len(texts)} embeddings in Pinecone")
    
    
    
    def query_embeddings(self, query, top_k=5, filter_dict=None):
        """Query Pinecone with a text input and get the most relevant results"""
        query_embedding = self.generate_query_embedding(query)
        
        
        # semantic search with filters - in pinecone
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        return results
    
    
    
    ## INDEX MANAGEMENT 
    
    def upsert(self, vectors):
        """Upsert a list of vectors to Pinecone"""
        self.index.upsert(vectors=vectors)
    
    def delete_all(self):
        """Delete all vectors in the index"""
        print("Deleting all vectors from index...")
        self.index.delete(delete_all=True)

    def describe_index(self):
        """Get basic info about the index"""
        return self.pc.describe_index(self.index_name)
