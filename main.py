import json
from dotenv import load_dotenv
import os
import asyncio

from store_scraper import StoreReviewsScraper
from embeddings_builder import HuggingFaceEmbeddingsBuilder
from clusters_generator import get_clustered_reviews, get_clustered_reviews_kmeans
from chroma_manager import ChromaManager


load_dotenv()
APP_URLS = [
    os.getenv("STARTING_APP_URL"),
    os.getenv("STARTING1_APP_URL"),
    os.getenv("STARTING2_APP_URL"),
    os.getenv("STARTING3_APP_URL"),
]
embedder = HuggingFaceEmbeddingsBuilder()
chroma_manager = ChromaManager()

async def scrape_reviews(scraper, urls, reviews_num):
    """Async wrapper for scraping reviews"""
    return await asyncio.to_thread(
        scraper.scrape,
        urls=urls,
        reviews_num=reviews_num
    )

async def main():
    # # Create scraper instance
    # tasks = []
    # scrapers_num = 4
    # for i in range(scrapers_num):
    #     scraper = StoreScraper(storage_manager=chroma_manager)
    #     tasks.append(
    #         scrape_reviews(scraper, [APP_URLS[i]], 10000),
    #     )
    
    # # Wait for both scraping tasks to complete
    # await asyncio.gather(*tasks)

    # Get the reviews and the embeddings
    reviews, embs = chroma_manager.get_all_docs_embs()

    # Generate clusters
    clusters = get_clustered_reviews_kmeans(reviews, embs)

    # Output report
    print(json.dumps(clusters, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())