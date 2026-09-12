import os

from dotenv import load_dotenv

# Load environment variables from a local .env file if present (git-ignored).
load_dotenv()


class Config:
    # Never hardcode real credentials. Set DATABASE_URL in your .env (see .env.example).
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "mysql+mysqlconnector://root:password@localhost:3306/company",
    )
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "80000"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    SUMMARIZE_THRESHOLD = int(os.getenv("SUMMARIZE_THRESHOLD", "80000"))

# Optionally, keep DocIngestConfig for backward compatibility or advanced use
class DocIngestConfig:
    def __init__(self, 
                 ollama_base_url: str = Config.OLLAMA_BASE_URL,
                 ollama_model: str = Config.OLLAMA_MODEL,
                 api_key: str = None):
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.api_key = api_key

    def to_dict(self):
        return {
            "ollama_base_url": self.ollama_base_url,
            "ollama_model": self.ollama_model,
            "api_key": self.api_key
        } 

AGENT_NAME = "data_insight"

PRT_WHITELISTED_TABLES = []

PERMANENT_STORAGE = "./storage"
SHARED_VOLUME_STORAGE = "./shared"
SQL_ROW_LIMIT = 1000
single_table_description_path = "/table_descriptions/"

# Additional config variables needed by main.py
ALLOWED_ORIGINS = ["*"]

DATA_INSIGHT_ABOUT_CONFIGURATION = {
    "service_name": "NL2SQL Data Insight Service",
    "url": "/business-solutions/sql-agent/run",
    "is_celery_endpoint": False,
    "description": "Natural language to SQL conversion service with chat interface",
    "user_type": "Business analysts and data professionals",
    "prompt_path": "./prompts/",
    "market": "Enterprise data analytics",
    "last_update": "2024-01-01"
}

def set_default_config():
    return {} 