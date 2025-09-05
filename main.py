import json
from dotenv import load_dotenv
import os

from scraper.store_scraper import StoreScraper
from embeddings_builder import HuggingFaceEmbeddingsBuilder
from clusters_generator import get_clustered_reviews, get_clustered_reviews_hdbscan
from chroma_manager import ChromaManager


load_dotenv()
STARTING_APP_URL = os.getenv("STARTING_APP_URL")
embedder = HuggingFaceEmbeddingsBuilder()
chroma_manager = ChromaManager()

if __name__ == "__main__":
    # Scrape negative reviews
    # scraper = StoreScraper(storage_manager=chroma_manager)
    # scraper.scrape(
    #     urls=[STARTING_APP_URL],
    #     reviews_num=10000
    # )

    # Get the reviews and the embeddings
    reviews, embs = chroma_manager.get_all_docs_embs()

    # Generate clusters
    clusters = get_clustered_reviews_hdbscan(reviews, embs)

    # Output report
    print(json.dumps(clusters, indent=4, ensure_ascii=False))