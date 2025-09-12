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
        self.embedder = embedder
        self.chroma_client = chromadb.PersistentClient(path="chroma_data")


    def save(self, text, metadata, collection):
        coll = self.chroma_client.get_or_create_collection(
            collection,
            embedding_function=self.embedder
        )

        results = coll.query(
            query_texts=[text],
            n_results=1
        )

        if (
            results['documents'] and 
            results['documents'][0] and 
            results['documents'][0][0] == text
        ):
            print(f"Document already exists ...")
            return False
        
        coll.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            metadatas=[metadata]
        )
        return True

    def get_all_docs_embs(self, collection):
        coll = self.chroma_client.get_or_create_collection(
            collection,
            embedding_function=self.embedder
        )

        results = coll.get(
            include=['embeddings', 'documents', 'metadatas']
        )
        texts = results.get('documents', [])
        embeddings = results.get('embeddings', [])
        metadatas = results.get('metadatas', [])
        return texts, embeddings, metadatas


if __name__ == "__main__":
    chroma_manager = ChromaManager()
    texts, embeddings, metadatas = chroma_manager.get_all_docs_embs("apps")
    print(len(texts))
    from clusters_generator import get_clustered_ranked_ideas
    get_clustered_ranked_ideas(embeddings, metadatas)