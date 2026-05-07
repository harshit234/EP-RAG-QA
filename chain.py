from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from embeddings_store import load_existing_store
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain(store_name="faiss_index"):
    """
    Create a RAG chain that answers questions using your documents
    """
    
    # Step 1: Load the vector store
    vector_store = load_existing_store(store_name)
    print("Vector store loaded")
    
    # Step 2: Create a retriever (searches the vector store)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("Retriever initialized (will fetch top 3 chunks)")
    
    # Step 3: Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
        convert_system_message_to_human=True
    )
    print("Gemini LLM initialized")
    
    # Step 4: Create custom prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer from the context, say "I don't have enough information in the documents to answer this."

Context:
{context}

Question: {question}

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    # Step 5: Create the RAG chain using LCEL
    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: retriever.invoke(x["question"])))
        | RunnablePassthrough.assign(
            answer=(
                lambda x: (PROMPT | llm | StrOutputParser()).invoke({
                    "context": format_docs(x["context"]),
                    "question": x["question"]
                })
            )
        )
    )
    
    print("RAG chain created successfully")
    
    return rag_chain

def answer_question(question, rag_chain):
    """
    Get answer for a question using the RAG chain
    """
    response = rag_chain.invoke({"question": question})
    
    answer = response["answer"]
    source_docs = response["context"]
    
    return answer, source_docs

# Test it
if __name__ == "__main__":
    # Create the RAG chain
    rag_chain = create_rag_chain()
    
    # Ask a question
    question = "What are the main points discussed?"
    print(f"\nQuestion: {question}")
    
    answer, sources = answer_question(question, rag_chain)
    
    print(f"\nAnswer:\n{answer}")
    
    print(f"\nSource documents used:")
    for i, doc in enumerate(sources, 1):
        print(f"\n{i}. Page {doc.metadata.get('page', 'Unknown')}:")
        safe_content = doc.page_content[:150].encode("ascii", "ignore").decode("ascii")
        print(f"   {safe_content}...")