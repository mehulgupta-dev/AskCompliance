import json
from langchain_community.vectorstores import Chroma
from embedding import dense_embeddings
from rag_backend.vector_store import vector_store_path

# Load your existing persisted Chroma store
vector_store = Chroma(
    persist_directory = vector_store_path,
    embedding_function = dense_embeddings
)

# Pull everything out
raw = vector_store.get(include=["documents", "metadatas"])

chunks = []
for doc_id, content, metadata in zip(raw["ids"], raw["documents"], raw["metadatas"]):
    chunks.append({
        "id": doc_id,
        "content": content,
        "metadata": metadata
    })

# Save to JSON
with open("chunks_export.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"Exported {len(chunks)} chunks to chunks_export.json")