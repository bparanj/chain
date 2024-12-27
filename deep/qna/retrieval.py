from pathlib import Path
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS  # Changed to FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
import os

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creds import openai

# pip install faiss-cpu

def check_openai_api_key():
    """Check if OPENAI_API_KEY is set and provide instructions if it's not."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OpenAI API key not found! Please set it by running:\n"
            "export OPENAI_API_KEY='your-api-key-here'\n"
            "or add it to your environment variables."
        )
    return api_key

# Check for API key before proceeding
api_key = check_openai_api_key()

# Get absolute path to the script's directory
script_dir = Path(__file__).resolve().parent

# Define the CSV file path relative to the script
csv_path = script_dir / '../data/data.csv'

print(f"Looking for CSV at: {csv_path.resolve()}")

# Load CSV documents
loader = CSVLoader(file_path=str(csv_path))
docs = loader.load()

# Print first document
print("First document:", docs[0])

# Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Test embedding
test_embed = embeddings.embed_query("Hi my name is Harrison")
print("Embedding length:", len(test_embed))
print("First 5 embedding values:", test_embed[:5])

# Create vector store using FAISS
db = FAISS.from_documents(docs, embeddings)

# Test similarity search
query = "Please suggest a shirt with sunblocking"
similar_docs = db.similarity_search(query)
print("Number of similar documents:", len(similar_docs))
print("First similar document:", similar_docs[0])

# Create retriever
retriever = db.as_retriever()

# Initialize ChatOpenAI
llm = ChatOpenAI(
    temperature=0.0,
    model="gpt-3.5-turbo",
    openai_api_key=api_key
)

# Method 1: Direct LLM call with concatenated documents
def get_docs_content(similar_docs):
    return "\n".join([doc.page_content for doc in similar_docs])

prompt_template = """Based on the following product information:
{context}

Question: {question}

Please provide the answer in markdown format."""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Create a chain for direct LLM calls
direct_chain = (
    {
        "context": lambda x: get_docs_content(similar_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Method 2: Updated RetrievalQA equivalent using LCEL
qa_prompt = ChatPromptTemplate.from_template("""
Answer the following question based on the provided context:

Context: {context}
Question: {question}

Provide your response in markdown format.
""")

# Create document chain
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt
)

# Create retrieval chain
retrieval_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | document_chain
)

# RAG chain with enhanced prompt
rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful shopping assistant. Based on the retrieved product information:

{context}

Please answer the following question in detail and in markdown format:
{question}

Make sure to include all relevant product details and format tables appropriately.
""")

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Example usage
if __name__ == "__main__":
    query = "Please summarize the customer reviews and list the main complaints and praises."

    print("\nMethod 1: Direct LLM Response")
    direct_response = direct_chain.invoke(query)
    print(direct_response)

    print("\nMethod 2: Retrieval Chain Response")
    qa_response = retrieval_chain.invoke(query)
    print(qa_response)

    print("\nRAG Chain Response")
    rag_response = rag_chain.invoke(query)
    print(rag_response)