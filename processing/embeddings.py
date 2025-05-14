# processing/embeddings.py
import os
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import uuid

# Load environment variables from .env file
load_dotenv()

class EmbeddingGenerator:
    def __init__(self, index_name=None, api_key=None, environment=None, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        self.environment = environment or os.getenv('PINECONE_ENVIRONMENT')
        self.index_name = index_name or os.getenv('PINECONE_INDEX_NAME') or "pagepal-shared"

        if not all([self.api_key, self.environment, self.index_name]):
            raise ValueError("Missing Pinecone credentials. Please check your .env file.")

        self.pc = Pinecone(api_key=self.api_key)
        indexes = [index.name for index in self.pc.list_indexes()]
        if self.index_name not in indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Created new Pinecone index: {self.index_name}")
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to Pinecone index: {self.index_name}")

    def generate_embeddings(self, texts):
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
        return self.model.encode(query)

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
            print(f"Storing vector with metadata: {metadata}")  # Debug
            vectors_to_upsert.append({
                "id": str(uuid.uuid4()),
                "values": embedding.tolist(),
                "metadata": metadata
            })
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i+batch_size]
            try:
                self.index.upsert(vectors=batch)
                print(f"Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1}")
            except Exception as e:
                print(f"Error upserting batch: {e}")
        print(f"Successfully stored {len(texts)} embeddings in Pinecone")

    def query_embeddings(self, query, top_k=5, filter_dict=None, include_values=False):
        """Query Pinecone with a text input and get the most relevant results"""
        query_embedding = self.generate_query_embedding(query)
        try:
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True,
                include_values=include_values,  # Add this line
                filter=filter_dict
            )
            print("Pinecone Results:", results)
            return results
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return {"matches": []}

    def upsert(self, vectors):
        self.index.upsert(vectors=vectors)

    def delete_all(self):
        print("Deleting all vectors from index...")
        self.index.delete(delete_all=True)

    def describe_index(self):
        return self.pc.describe_index(self.index_name)
