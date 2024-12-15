## Prompt

Update the code below to work with OpenAI version 1.57 and LangChain 0.3:

All the documents in llm-app-dev is upgraded to latest by Claude.

## Tasks

- Continue to test the following code examples:

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

# Initialize the chat model
llm = ChatOpenAI(
    temperature=0.9,
    model="gpt-3.5-turbo"  # Replace with your preferred model
)

# Prompt template 1
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)

chain_one = LLMChain(llm=llm, prompt=first_prompt)

# Prompt template 2
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20-word description for the following company: {company_name}"
)

chain_two = LLMChain(llm=llm, prompt=second_prompt)

# Combine chains
overall_simple_chain = SimpleSequentialChain(
    chains=[chain_one, chain_two],
    verbose=True
)

# Run the combined chain
product = "Queen Size Sheet Set"  # Example product
response = overall_simple_chain.invoke({"product": product})

# Debugging response structure
print(response)  # Check the full response structure

# Accessing final output
if "output" in response:
    print(response["output"])  # Replace 'output' with the correct key after debugging
else:
    print("Response structure unexpected. Debug the output.")

----

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
import pandas as pd

# Initialize the chat model
llm = ChatOpenAI(
    temperature=0.9,
    model="gpt-3.5-turbo"  # Replace with your preferred model
)

# Chain 1: Translate to English
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to English:\n\n{Review}"
)
chain_one = LLMChain(
    llm=llm,
    prompt=first_prompt,
    output_key="English_Review"
)

# Chain 2: Summarize
second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:\n\n{English_Review}"
)
chain_two = LLMChain(
    llm=llm,
    prompt=second_prompt,
    output_key="summary"
)

# Chain 3: Detect Language
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)
chain_three = LLMChain(
    llm=llm,
    prompt=third_prompt,
    output_key="language"
)

# Chain 4: Generate Follow-up
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow-up response to the following summary in the specified language:\n\nSummary: {summary}\n\nLanguage: {language}"
)
chain_four = LLMChain(
    llm=llm,
    prompt=fourth_prompt,
    output_key="followup_message"
)

# Combine all chains
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary", "followup_message"],
    verbose=True
)

# Run the chain with a review
review = df.Review[5]  # Assuming df is your DataFrame
response = overall_chain.invoke({"Review": review})

# Access results
print("English Review:", response["English_Review"])
print("Summary:", response["summary"])
print("Follow-up Message:", response["followup_message"])

----

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

# Define templates
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""

math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts,\
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""

computerscience_template = """You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity.

Here is a question:
{input}"""

# Define prompt information
prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template
    },
    {
        "name": "History",
        "description": "Good for answering history questions",
        "prompt_template": history_template
    },
    {
        "name": "computer science",
        "description": "Good for answering computer science questions",
        "prompt_template": computerscience_template
    }
]

# Initialize the LLM
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"  # Replace with your preferred model
)

# Create destination chains
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

# Create destinations string
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# Create default chain
default_prompt = PromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

# Define router template
MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

# Create router chain
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# Create multi-prompt chain
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True
)

# Example usage
questions = [
    "What is black body radiation?",
    "what is 2 + 2",
    "Why does every cell in our body contain DNA?"
]

for question in questions:
    response = chain.invoke({"input": question})
    print(f"\nQuestion: {question}")
    print(f"Response: {response['text']}")


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

