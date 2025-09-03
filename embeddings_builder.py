from openai import OpenAI
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class OpenAIEmbeddingsBuilder:
    def __init__(self):
        pass

    def get_embeddings(self, text_items: list) -> list:
        res = client.embeddings.create(
            input=text_items,
            model="text-embedding-3-small"
        )
        return [item.embedding for item in res.data]
    

class HuggingFaceEmbeddingsBuilder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_embeddings(self, text_items: list) -> list:
        return self.model.encode(text_items)