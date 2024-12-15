## Prompt

Update the code below to work with OpenAI version 1.57 and LangChain 0.3:

All the documents in llm-app-dev is upgraded to latest by Claude.

## Tasks

- Look at retrieval.py and compare the output and setps with the first video in Question and Answer section.
- Continue to test the following code examples:

import pandas as pd
df = pd.read_csv('Data.csv')
```


```python
df.head()
```

----

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.runnables import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from IPython.display import display, Markdown

# Load the CSV file
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
documents = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splits = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = DocArrayInMemorySearch.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever()

# Create the prompt template
template = """Answer the following question based only on the provided context:

Context: {context}

Question: {question}

Answer in markdown format."""

prompt = PromptTemplate.from_template(template)

# Initialize the model
model = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"
)

# Create the RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Example query
query = "Please list all your shirts with sun protection in a table in markdown and summarize each one."
response = chain.invoke({"context": retriever.get_relevant_documents(query), "question": query})

# Display the response
display(Markdown(response))

----

from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from IPython.display import display, Markdown

# Load CSV documents
file = 'OutdoorClothingCatalog_1000.csv'  # Replace with your file path
loader = CSVLoader(file_path=file)
docs = loader.load()

# Print first document
print("First document:", docs[0])

# Create embeddings
embeddings = OpenAIEmbeddings()

# Test embedding
test_embed = embeddings.embed_query("Hi my name is Harrison")
print("Embedding length:", len(test_embed))
print("First 5 embedding values:", test_embed[:5])

# Create vector store
db = DocArrayInMemorySearch.from_documents(
    docs,
    embeddings
)

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
    model="gpt-3.5-turbo"  # Replace with your preferred model
)

# Method 1: Direct LLM call with concatenated documents
qdocs = "".join([doc.page_content for doc in similar_docs])
prompt_template = """Based on the following product information:
{context}

Question: {question}

Please provide the answer in markdown format."""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Create a chain for direct LLM calls
direct_chain = (
    {"context": lambda x: qdocs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Method 2: RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
    return_source_documents=True
)

# Example query
query = "Please list all your shirts with sun protection in a table in markdown and summarize each one."

# Get responses using different methods
print("\nMethod 1: Direct LLM Response")
direct_response = direct_chain.invoke(query)
display(Markdown(direct_response))

print("\nMethod 2: RetrievalQA Response")
qa_response = qa_chain.invoke({"query": query})
display(Markdown(qa_response['result']))

# Optional: Create a RAG chain combining both approaches
combined_prompt = ChatPromptTemplate.from_template("""
You are a helpful shopping assistant. Based on the retrieved product information:

{context}

Please answer the following question in detail and in markdown format:
{question}

Make sure to include all relevant product details and format tables appropriately.
""")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | combined_prompt
    | llm
    | StrOutputParser()
)

print("\nRAG Chain Response")
rag_response = rag_chain.invoke(query)
display(Markdown(rag_response))

----

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

# Load the CSV file
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

# Create text splitter
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator="\n"
)

# Split documents
splits = text_splitter.split_documents(data)

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = DocArrayInMemorySearch.from_documents(
    splits,
    embeddings
)

# Initialize the chat model
llm = ChatOpenAI(
    temperature=0.0,
    model="gpt-3.5-turbo"  # Replace with your preferred model
)

# Create custom prompt template
prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: """

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "prompt": PROMPT
    },
    return_source_documents=True
)

# Example documents for reference
print("Example Document 1:")
print(data[10])
print("\nExample Document 2:")
print(data[11])

# Test examples
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty 850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]

# Function to run tests
def run_qa_tests(qa_chain, test_examples):
    results = []
    for i, example in enumerate(test_examples):
        print(f"\nRunning test {i + 1}:")
        print(f"Query: {example['query']}")
        print(f"Expected Answer: {example['answer']}")

        response = qa_chain({"query": example['query']})
        print(f"Actual Answer: {response['result']}")

        results.append({
            "query": example['query'],
            "expected": example['answer'],
            "actual": response['result'],
            "source_docs": response['source_documents']
        })
    return results

# Run the tests
test_results = run_qa_tests(qa, examples)

# Print detailed results
print("\nDetailed Test Results:")
for i, result in enumerate(test_results):
    print(f"\nTest {i + 1}:")
    print(f"Query: {result['query']}")
    print(f"Expected: {result['expected']}")
    print(f"Actual: {result['actual']}")
    print("Source Documents Used:")
    for j, doc in enumerate(result['source_docs']):
        print(f"\nDocument {j + 1}:")
        print(doc.page_content[:200] + "...")  # Print first 200 characters of each source

----

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

# Load the CSV file
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

# Check if data is loaded
if not data:
    raise ValueError("No data loaded. Ensure the CSV file exists and is not empty.")

# Create text splitter
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator="\n"
)

# Split documents
splits = text_splitter.split_documents(data)

# Check if documents are split
if not splits:
    raise ValueError("No document splits created. Check the input data format and text splitter configuration.")

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = DocArrayInMemorySearch.from_documents(
    splits,
    embeddings
)

# Initialize the chat model
llm = ChatOpenAI(
    temperature=0.0,
    model="gpt-3.5-turbo"  # Replace with your preferred model
)

# Create custom prompt template
prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: """

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "prompt": PROMPT
    },
    return_source_documents=True
)

# Example documents for reference
if len(data) > 10:
    print("Example Document 1:")
    print(data[10])
if len(data) > 11:
    print("\nExample Document 2:")
    print(data[11])

# Test examples
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty 850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]

# Function to run tests
def run_qa_tests(qa_chain, test_examples):
    results = []
    for i, example in enumerate(test_examples):
        print(f"\nRunning test {i + 1}:")
        print(f"Query: {example['query']}")
        print(f"Expected Answer: {example['answer']}")

        response = qa_chain({"query": example['query']})
        print(f"Actual Answer: {response['result']}")

        results.append({
            "query": example['query'],
            "expected": example['answer'],
            "actual": response['result'],
            "source_docs": response['source_documents']
        })
    return results

# Run the tests
test_results = run_qa_tests(qa, examples)

# Print detailed results
print("\nDetailed Test Results:")
for i, result in enumerate(test_results):
    print(f"\nTest {i + 1}:")
    print(f"Query: {result['query']}")
    print(f"Expected: {result['expected']}")
    print(f"Actual: {result['actual']}")
    print("Source Documents Used:")
    for j, doc in enumerate(result['source_docs']):
        print(f"\nDocument {j + 1}:")
        print(doc.page_content[:200] + "...")  # Print first 200 characters of each source


----

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Load the CSV file
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

# Check if data is loaded
if not data:
    raise ValueError("No data loaded. Ensure the CSV file exists and is not empty.")

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n", ". ", " "]
)

# Split documents
splits = text_splitter.split_documents(data)

# Check if documents are split
if not splits:
    raise ValueError("No document splits created. Check the input data format and text splitter configuration.")

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = DocArrayInMemorySearch.from_documents(
    splits,
    embeddings
)

# Initialize the chat model
llm = ChatOpenAI(
    temperature=0.0,
    model="gpt-3.5-turbo"  # Replace with your preferred model
)

# Create custom prompt template
prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: """

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "prompt": PROMPT
    },
    return_source_documents=True
)

# Example documents for reference
if len(data) > 10:
    print("Example Document 1:")
    print(data[10])
if len(data) > 11:
    print("\nExample Document 2:")
    print(data[11])

# Test examples
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty 850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]

# Function to run tests
def run_qa_tests(qa_chain, test_examples):
    results = []
    for i, example in enumerate(test_examples):
        print(f"\nRunning test {i + 1}:")
        print(f"Query: {example['query']}")
        print(f"Expected Answer: {example['answer']}")

        response = qa_chain({"query": example['query']})
        print(f"Actual Answer: {response['result']}")

        results.append({
            "query": example['query'],
            "expected": example['answer'],
            "actual": response['result'],
            "source_docs": response['source_documents']
        })
    return results

# Run the tests
test_results = run_qa_tests(qa, examples)

# Print detailed results
print("\nDetailed Test Results:")
for i, result in enumerate(test_results):
    print(f"\nTest {i + 1}:")
    print(f"Query: {result['query']}")
    print(f"Expected: {result['expected']}")
    print(f"Actual: {result['actual']}")
    print("Source Documents Used:")
    for j, doc in enumerate(result['source_docs']):
        print(f"\nDocument {j + 1}:")
        print(doc.page_content[:200] + "...")  # Print first 200 characters of each source

