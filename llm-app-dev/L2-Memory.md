# LangChain: Memory

## Outline
* ConversationBufferMemory
* ConversationBufferWindowMemory
* ConversationTokenBufferMemory
* ConversationSummaryMemory

## ConversationBufferMemory


```python
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import warnings
warnings.filterwarnings('ignore')
```


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
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
```


```python
llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=True
)
```


```python
conversation.predict(input="Hi, my name is Andrew")
```


```python
conversation.predict(input="What is 1+1?")
```


```python
conversation.predict(input="What is my name?")
```


```python
print(memory.buffer)
```


```python
memory.load_memory_variables({})
```


```python
memory = ConversationBufferMemory()
```


```python
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
```


```python
print(memory.buffer)
```


```python
memory.load_memory_variables({})
```


```python
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
```


```python
memory.load_memory_variables({})
```

Here's the updated code for OpenAI 1.57 and LangChain 0.3:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize the model - no need for llm_model parameter
llm = ChatOpenAI(temperature=0.0)

# Initialize memory
memory = ConversationBufferMemory(return_messages=True)

# Create conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Make predictions
conversation.predict(input="Hi, my name is Andrew")
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")

# View memory contents
print(memory.buffer)

# Load memory variables
memory.load_memory_variables({})

# Create new memory instance
memory = ConversationBufferMemory(return_messages=True)

# Save context to memory
memory.save_context({"input": "Hi"},
                   {"output": "What's up"})

# View buffer
print(memory.buffer)

# Load memory variables
memory.load_memory_variables({})

# Save more context
memory.save_context({"input": "Not much, just hanging"},
                   {"output": "Cool"})

# Load final memory state
memory.load_memory_variables({})
```

changes:
1. Changed `langchain.chat_models` to `langchain_openai`
2. Added `return_messages=True` to ConversationBufferMemory for compatibility with newer versions
3. Removed the `llm_model` parameter as it's not needed (it will use gpt-3.5-turbo by default)
4. Added import for `MessagesPlaceholder` from `langchain_core.prompts`

Make sure you have the required packages installed:
```bash
pip install langchain-openai langchain-core
```

And your OpenAI API key set in environment variables:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## ConversationBufferWindowMemory


```python
from langchain.memory import ConversationBufferWindowMemory
```


```python
memory = ConversationBufferWindowMemory(k=1)
```


```python
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})

```


```python
memory.load_memory_variables({})
```


```python
llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=False
)
```


```python
conversation.predict(input="Hi, my name is Andrew")
```


```python
conversation.predict(input="What is 1+1?")
```


```python
conversation.predict(input="What is my name?")
```

Here's the updated code for OpenAI 1.57 and LangChain 0.3:

```python
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

# Initialize window memory with k=1
memory = ConversationBufferWindowMemory(k=1, return_messages=True)

# Save contexts
memory.save_context({"input": "Hi"},
                   {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                   {"output": "Cool"})

# Load memory variables
memory.load_memory_variables({})

# Initialize chat and conversation with window memory
llm = ChatOpenAI(temperature=0.0)  # removed llm_model parameter
memory = ConversationBufferWindowMemory(k=1, return_messages=True)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# Make predictions
conversation.predict(input="Hi, my name is Andrew")
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")
```

Key changes:
1. Changed import to `from langchain_openai import ChatOpenAI`
2. Added `return_messages=True` to ConversationBufferWindowMemory instances
3. Removed the `llm_model` parameter as it's not needed

Make sure you have installed:
```bash
pip install langchain-openai
```

## ConversationTokenBufferMemory


```python
#!pip install tiktoken
```


```python
from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import OpenAI
llm = ChatOpenAI(temperature=0.0, model=llm_model)
```


```python
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"},
                    {"output": "Charming!"})
```


```python
memory.load_memory_variables({})
```

## ConversationSummaryMemory


```python
from langchain.memory import ConversationSummaryBufferMemory
```


```python
# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"},
                    {"output": f"{schedule}"})
```


```python
memory.load_memory_variables({})
```


```python
conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=True
)
```


```python
conversation.predict(input="What would be a good demo to show?")
```


```python
memory.load_memory_variables({})
```

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import os

# Initialize the LLM
llm = ChatOpenAI(
    temperature=0.7,
    model="gpt-3.5-turbo"  # or "gpt-4" if you prefer
)

# Create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

# Initialize memory
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=100,
    return_messages=True  # Important: This is required in newer versions
)

# Save contexts
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context(
    {"input": "Not much, just hanging"},
    {"output": "Cool"}
)
memory.save_context(
    {"input": "What is on the schedule today?"},
    {"output": f"{schedule}"}
)

# Load memory variables
memory_variables = memory.load_memory_variables({})

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Create conversation chain
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# Make a prediction
response = conversation.invoke({"input": "What would be a good demo to show?"})
print(response['text'])

# Load final memory state
final_memory = memory.load_memory_variables({})
