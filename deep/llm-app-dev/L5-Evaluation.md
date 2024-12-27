# LangChain: Evaluation

## Outline:

* Example generation
* Manual evaluation (and debuging)
* LLM-assisted evaluation
* LangChain evaluation platform


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

## Create our QandA application


```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
```


```python
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()
```


```python
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])
```


```python
llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)
```

### Coming up with test datapoints


```python
data[10]
```


```python
data[11]
```

### Hard-coded examples


```python
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]
```

from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate

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
        "document_separator": "<<<<>>>>>",
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
        
        response = qa_chain.invoke({"query": example['query']})
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



### LLM-Generated examples


```python
from langchain.evaluation.qa import QAGenerateChain

```


```python
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))
```


```python
# the warning below can be safely ignored
```


```python
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)
```


```python
new_examples[0]
```


```python
data[0]
```

### Combine examples


```python
examples += new_examples
```


```python
qa.run(examples[0]["query"])
```

## Manual Evaluation


```python
import langchain
langchain.debug = True
```


```python
qa.run(examples[0]["query"])
```


```python
# Turn off the debug mode
langchain.debug = False
```

## LLM assisted evaluation


```python
predictions = qa.apply(examples)
```


```python
from langchain.evaluation.qa import QAEvalChain
```


```python
llm = ChatOpenAI(temperature=0, model=llm_model)
eval_chain = QAEvalChain.from_llm(llm)
```


```python
graded_outputs = eval_chain.evaluate(examples, predictions)
```


```python
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()
```


```python
graded_outputs[0]
```

----

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.evaluation import StringEvaluator
from langchain_core.runnables import RunnablePassthrough
import json
import logging

# Initialize the chat model
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"  # Replace with your preferred model
)

# Create a QA generation prompt
qa_gen_prompt = PromptTemplate(
    template="""Given the following document, generate a question and answer pair.
The question should be detailed and specific to the document content.
The answer should be correct and concise.

Document: {doc}

Output your response in the following format:
```json
{{
    "query": "your question here",
    "answer": "your answer here"
}}
```""",
    input_variables=["doc"]
)

# Create QA generation chain
qa_gen_chain = LLMChain(
    llm=llm,
    prompt=qa_gen_prompt,
    output_key="qa_pair"
)

def parse_qa_response(response):
    try:
        # Extract JSON from potential markdown code block
        json_str = response['qa_pair']
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        return json.loads(json_str)
    except Exception as e:
        logging.error(f"Error parsing QA response: {e}")
        return {"query": "", "answer": ""}

# Generate new examples
def generate_qa_examples(documents):
    examples = []
    for doc in documents[:5]:  # Using first 5 documents
        response = qa_gen_chain.invoke({"doc": doc.page_content})
        qa_pair = parse_qa_response(response)
        if qa_pair["query"] and qa_pair["answer"]:
            examples.append(qa_pair)
    return examples

# Generate new examples
new_examples = generate_qa_examples(data)
print("First generated example:", new_examples[0])
print("\nSource document:", data[0])

# Combine examples
examples += new_examples

# Create evaluation prompt
eval_prompt = PromptTemplate(
    template="""You are evaluating a question answering system.
Please assess if the predicted answer correctly answers the question given the real answer as reference.

Question: {query}
Real Answer: {answer}
Predicted Answer: {result}

Evaluate the predicted answer on the following criteria:
1. Correctness: Is the information accurate compared to the real answer?
2. Completeness: Does it cover all aspects mentioned in the real answer?
3. Relevance: Is it directly addressing the question?

Output your evaluation as a single word: "CORRECT" or "INCORRECT"

Evaluation:""",
    input_variables=["query", "answer", "result"]
)

# Create evaluation chain
eval_chain = LLMChain(
    llm=llm,
    prompt=eval_prompt,
    output_key="evaluation"
)

# Function to run evaluations
def evaluate_qa_predictions(qa_chain, examples):
    results = []
    for example in examples:
        # Get prediction
        prediction = qa_chain.invoke({"query": example["query"]})
        
        # Evaluate prediction
        evaluation = eval_chain.invoke({
            "query": example["query"],
            "answer": example["answer"],
            "result": prediction["result"]
        })
        
        results.append({
            "query": example["query"],
            "real_answer": example["answer"],
            "predicted_answer": prediction["result"],
            "evaluation": evaluation["evaluation"].strip(),
            "source_documents": prediction.get("source_documents", [])
        })
    return results

# Enable debug mode for detailed chain execution
import langchain
langchain.debug = True

# Run evaluation
evaluation_results = evaluate_qa_predictions(qa, examples)

# Disable debug mode
langchain.debug = False

# Print evaluation results
print("\nEvaluation Results:")
for i, result in enumerate(evaluation_results):
    print(f"\nExample {i + 1}:")
    print(f"Question: {result['query']}")
    print(f"Real Answer: {result['real_answer']}")
    print(f"Predicted Answer: {result['predicted_answer']}")
    print(f"Evaluation: {result['evaluation']}")
    if result['source_documents']:
        print("\nSource Documents Used:")
        for j, doc in enumerate(result['source_documents']):
            print(f"\nDocument {j + 1}:")
            print(doc.page_content[:200] + "...")  # Print first 200 characters

# Calculate overall performance
correct_count = sum(1 for r in evaluation_results if r['evaluation'] == "CORRECT")
total_count = len(evaluation_results)
accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

print(f"\nOverall Accuracy: {accuracy:.2f}%")
print(f"Correct: {correct_count}/{total_count}")
