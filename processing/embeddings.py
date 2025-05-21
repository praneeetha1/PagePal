import os
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import uuid
import time

class EmbeddingGenerator:
    def __init__(self, index_name=None, api_key=None, environment=None, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self.environment = environment or os.environ.get("PINECONE_ENVIRONMENT")
        self.index_name = index_name or "pagepal-shared"

        if not all([self.api_key, self.environment]):
            raise ValueError("Missing Pinecone credentials")

        self.pc = Pinecone(api_key=self.api_key)
        self._init_index()

    def _init_index(self):
        existing_indexes = self.pc.list_indexes().names()
        
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Created index: {self.index_name}")
            self._wait_for_index_ready()

        self.index = self.pc.Index(self.index_name)
        print(f"Connected to index: {self.index_name}")

    def _wait_for_index_ready(self, timeout=300):
        start = time.time()
        while time.time() - start < timeout:
            status = self.pc.describe_index(self.index_name).status
            if status.get("ready"):
                return
            time.sleep(5)
        raise TimeoutError("Index creation timed out")

    def generate_embeddings(self, texts):
        if not texts:
            return np.array([])
            
        return np.array([
            self.model.encode(batch)
            for batch in self._batch(texts, 32)
        ]).reshape(-1, 384)

    def store_embeddings(self, texts, metadata_list=None):
        if not texts:
            return

        embeddings = self.generate_embeddings(texts)
        vectors = [{
            "id": str(uuid.uuid4()),
            "values": emb.tolist(),
            "metadata": {**(metadata_list[i] if metadata_list else {}), "text": text}
        } for i, (text, emb) in enumerate(zip(texts, embeddings))]

        for batch in self._batch(vectors, 100):
            self.index.upsert(vectors=batch)

    def query_embeddings(self, query, top_k=5, filter_dict=None):
        query_emb = self.model.encode(query).tolist()
        return self.index.query(
            vector=query_emb,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )

    @staticmethod
    def _batch(items, size):
        return [items[i:i+size] for i in range(0, len(items), size)]
