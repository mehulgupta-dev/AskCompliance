from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

dense_embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-large"
)