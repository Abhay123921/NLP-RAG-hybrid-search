import os
import pandas as pd
from src.config import DATA_PATH, TEXT_COLUMN, TITLE_COLUMN

print("Looking for file at:", DATA_PATH)
print("File exists?", os.path.exists(DATA_PATH))

def load_data():
    df = pd.read_csv(DATA_PATH)

    documents = []
    for i, row in df.iterrows():
        documents.append({
            "id": i,
            "title": str(row.get(TITLE_COLUMN, "")),
            "text": str(row[TEXT_COLUMN]),
            "metadata": {}
        })

    return documents