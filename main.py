import json
from dotenv import load_dotenv
import os
import asyncio

from scraper.store_scraper import StoreScraper
from embeddings_builder import HuggingFaceEmbeddingsBuilder
from clusters_generator import get_clustered_reviews
from chroma_manager import ChromaManager


load_dotenv()
STARTING_APP_URL = os.getenv("STARTING_APP_URL")
STARTING1_APP_URL = os.getenv("STARTING1_APP_URL")
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
    # Create scraper instance
    scraper = StoreScraper(storage_manager=chroma_manager)
    scraper1 = StoreScraper(storage_manager=chroma_manager)
    
    # Run scraper 2 times concurrently
    tasks = [
        scrape_reviews(scraper, [STARTING_APP_URL], 10000),
        scrape_reviews(scraper1, [STARTING1_APP_URL], 10000)
    ]
    
    # Wait for both scraping tasks to complete
    await asyncio.gather(*tasks)

    # Get the reviews and the embeddings
    reviews, embs = chroma_manager.get_all_docs_embs()

    # Generate clusters
    clusters = get_clustered_reviews(reviews, embs)

    # Output report
    print(json.dumps(clusters, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())