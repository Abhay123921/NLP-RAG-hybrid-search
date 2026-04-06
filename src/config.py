import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "processed", "documents.json")

TEXT_COLUMN = "content"
TITLE_COLUMN = "title"

CHUNK_SIZE = 200
USE_CHUNKING = True
REMOVE_DUPLICATES = True