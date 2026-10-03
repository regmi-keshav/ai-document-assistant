from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import ChromaVectorStore

from app.config import DOCUMENTS_PATH
from app.rag.loader import PDFDocumentLoader
from app.rag.chunker import TextChunker

def main():

    # --------------------------------------------------
    # 1. Load documents
    # --------------------------------------------------

    loader = PDFDocumentLoader(DOCUMENTS_PATH)

    documents = loader.load()

    print(f"Loaded documents: {len(documents)}")

    # --------------------------------------------------
    # 2. Chunk documents
    # --------------------------------------------------

    chunker = TextChunker()

    chunks = chunker.chunk_documents(documents)

    print(f"Generated chunks: {len(chunks)}")

    # --------------------------------------------------
    # 3. Generate embeddings
    # --------------------------------------------------

    embedding_service = EmbeddingService()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_service.embed(texts)

    print(f"Generated embeddings: {len(embeddings)}")
    print(
        f"Embedding dimension: "
        f"{embedding_service.dimension}"
    )

    # --------------------------------------------------
    # 4. Store in ChromaDB
    # --------------------------------------------------

    vector_store = ChromaVectorStore()
    vector_store.clear()

    ids = [
        f"{chunk['source']}-{chunk['page']}-{index}"
        for index, chunk in enumerate(chunks)
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
        }
        for chunk in chunks
    ]

    vector_store.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(
        f"Documents stored in ChromaDB: "
        f"{vector_store.count()}"
    )


if __name__ == "__main__":
    main()