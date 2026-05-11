import os
from dotenv import load_dotenv

load_dotenv()

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://opensearch:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "nhs_conditions")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
NHS_CONDITIONS_URL = "https://www.nhs.uk/health-a-to-z/conditions/"
