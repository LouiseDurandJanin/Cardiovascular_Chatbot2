from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)
scroll_result = client.scroll(
    collection_name="vector_db",
    limit=100,
    with_payload=True
)

# Afficher les sources des vecteurs

for point in scroll_result[0]:
    source = point.payload.get("metadata", {}).get("source")
    if source:
        print(source) 
