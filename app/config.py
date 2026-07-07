import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "./data")
MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", "5"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

DOCUMENT_TTL_MINUTES = int(os.getenv("DOCUMENT_TTL_MINUTES", "20"))
DOCUMENT_TTL_SECONDS = DOCUMENT_TTL_MINUTES * 60
PURGE_INTERVAL_SECONDS = int(os.getenv("PURGE_INTERVAL_SECONDS", "60"))

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)