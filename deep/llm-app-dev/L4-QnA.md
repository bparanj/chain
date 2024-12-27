# LangChain: Q&A over Documents

An example might be a tool that would allow you to query a product catalog for items of interest.


```python
#pip install --upgrade langchain
```


```python
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
```

Note: LLM's do not always produce the same results. When executing the code in your notebook, you may get slightly different answers that those in the video.


```python
# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"
```


```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.llms import OpenAI
```


```python
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
```


```python
from langchain.indexes import VectorstoreIndexCreator
```


```python
#pip install docarray
```


```python
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])
```


```python
query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."
```

**Note**:
- The notebook uses `langchain==0.0.179` and `openai==0.27.7`
- For these library versions, `VectorstoreIndexCreator` uses `text-davinci-003` as the base model, which has been deprecated since 1 January 2024.
- The replacement model, `gpt-3.5-turbo-instruct` will be used instead for the `query`.
- The `response` format might be different than the video because of this replacement model.


```python
llm_replacement_model = OpenAI(temperature=0, 
                               model='gpt-3.5-turbo-instruct')

response = index.query(query, 
                       llm = llm_replacement_model)
```


```python
display(Markdown(response))
```

from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
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

prompt = ChatPromptTemplate.from_template(template)

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
response = chain.invoke(query)

# Display the response
display(Markdown(response))

## Step By Step


```python
from langchain.document_loaders import CSVLoader
loader = CSVLoader(file_path=file)
```


```python
docs = loader.load()
```


```python
docs[0]
```


```python
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
```


```python
embed = embeddings.embed_query("Hi my name is Harrison")
```


```python
print(len(embed))
```


```python
print(embed[:5])
```


```python
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)
```


```python
query = "Please suggest a shirt with sunblocking"
```


```python
docs = db.similarity_search(query)
```


```python
len(docs)
```


```python
docs[0]
```


```python
retriever = db.as_retriever()
```


```python
llm = ChatOpenAI(temperature = 0.0, model=llm_model)
```


```python
qdocs = "".join([docs[i].page_content for i in range(len(docs))])

```


```python
response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 

```


```python
display(Markdown(response))
```


```python
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
```


```python
query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."
```


```python
response = qa_stuff.run(query)
```


```python
display(Markdown(response))
```


```python
response = index.query(query, llm=llm)
```


```python
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])
```

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
