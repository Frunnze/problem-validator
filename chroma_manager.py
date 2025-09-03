
import chromadb
from chromadb.utils import embedding_functions
import uuid


class ChromaManager:
    def __init__(self, embedder=None):
        if not embedder:
            embedder = (
                embedding_functions
                .SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            )
        self.chroma_client = chromadb.PersistentClient(path="chroma_data")
        self.reviews_coll = self.chroma_client.get_or_create_collection(
            "reviews",
            embedding_function=embedder
        )

    def save(self, text, metadata):
        # Check if document already exists by querying for exact text match
        results = self.reviews_coll.query(
            query_texts=[text],
            n_results=1
        )
        
        # If we get results and the first result is an exact match
        if results['documents'] and results['documents'][0]:
            if results['documents'][0][0] == text:
                print(f"Document already exists ...")
                return False
        
        self.reviews_coll.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            metadatas=[metadata]
        )
        return True

    def get_all_docs_embs(self):
        results = self.reviews_coll.get(
            include=['embeddings', 'documents', 'metadatas']
        )
        texts = results.get('documents', [])
        embeddings = results.get('embeddings', [])
        return texts, embeddings


if __name__ == "__main__":
    chroma_manager = ChromaManager()
    chroma_manager.save(text="Hello", metadata={"review": 1})
    print(chroma_manager.get_all_docs_embs())