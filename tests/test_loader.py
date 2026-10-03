from app.rag.loader import PDFDocumentLoader


loader = PDFDocumentLoader("data/documents")

documents = loader.load()

for document in documents:
    print(f"\nSOURCE: {document['source']}")

    for page in document["pages"][:2]:
        print(f"PAGE: {page['page']}")
        print(page["text"][:500])