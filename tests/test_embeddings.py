from app.rag.embeddings import EmbeddingModel


embedding_model = EmbeddingModel()

texts = [
    "Retrieval-Augmented Generation combines retrieval with language generation.",
    "Fine-tuning updates the parameters of a language model.",
]

embeddings = embedding_model.embed(
    texts
)

print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimension: {len(embeddings[0])}")
print(f"First 5 values: {embeddings[0][:5]}")