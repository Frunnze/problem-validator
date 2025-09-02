from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(text_items: list) -> list:
    res = client.embeddings.create(
        input=text_items,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in res.data]