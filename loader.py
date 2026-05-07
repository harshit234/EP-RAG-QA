from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # Each chunk = 1000 characters
        chunk_overlap=200,        # 200 char overlap between chunks
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    return chunks, documents


if __name__ == "__main__":
    pdf_path = "sample_docs/EJ1200789.pdf"
    chunks, docs = load_and_chunk_pdf(pdf_path)
    
    print(f"Total documents: {len(docs)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"\nFirst chunk preview:")
    print(chunks[0].page_content[:500])