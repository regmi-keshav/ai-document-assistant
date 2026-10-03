from app.rag.loader import PDFDocumentLoader
from app.rag.chunker import TextChunker


loader = PDFDocumentLoader("data/documents")
documents = loader.load()

chunker = TextChunker(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = chunker.chunk_documents(documents)

print(f"Documents: {len(documents)}")
print(f"Chunks: {len(chunks)}")

for i, chunk in enumerate(chunks[:5], start=1):
    print(f"\n--- CHUNK {i} ---")
    print(f"Source: {chunk['source']}")
    print(f"Page: {chunk['page']}")
    print(f"Characters: {len(chunk['text'])}")
    print(chunk["text"])