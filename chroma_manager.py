import chromadb
from chromadb.utils import embedding_functions
import uuid
from copy import deepcopy


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
            n_results=1,
            include=["documents", "distances"]
        )

        if results['distances'][0][0] == 0:
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

    # def count_all(self):
    #     try:
    #         results = self.reviews_coll.get()
    #         return len(results.get('documents', []))
    #     except Exception as e:
    #         print(f"Error counting reviews: {e}")
    #         return 0

    # def get_similar_reviews(self, text):
    #     results = self.reviews_coll.query(
    #         query_texts=[text],
    #         n_results=100,
    #         include=["documents"]
    #     )
    #     return results

if __name__ == "__main__":
    from sklearn.cluster import DBSCAN

    chroma_manager = ChromaManager()
    texts, embeddings, metadatas = chroma_manager.get_all_docs_embs("apps")
    print(len(metadatas))

    db = DBSCAN(eps=0.2, min_samples=2, metric="cosine").fit(embeddings)
    labels = db.labels_

    clusters = {}
    for label, meta in zip(labels, metadatas):
        # if int(label) == -1:
        #     continue
        if int(label) not in clusters:
            clusters[int(label)] = deepcopy(meta)
        clusters[int(label)]["sum_rating"] = clusters[int(label)].get("sum_rating", 0) + meta["rating"]
        clusters[int(label)]["sim_apps_num"] = clusters[int(label)].get("sim_apps_num", 0) + 1

    clusters = sorted(list(clusters.values()), key=lambda x: x["sum_rating"]/x["sim_apps_num"])
    import json
    print(json.dumps(clusters, indent=4))