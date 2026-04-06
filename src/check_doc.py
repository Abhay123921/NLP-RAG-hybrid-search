import json

# Load processed documents
with open("processed/documents.json", "r") as f:
    docs = json.load(f)

print("Total documents:", len(docs))
print("\nSample documents:\n")

# Print first 20 docs
for d in docs[:20]:
    print(f"ID: {d['id']}")
    print(f"Text: {d['text'][:80]}...")
    print("-" * 50)