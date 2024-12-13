The basics folder consists of setup instructions and programs to check the OpenAI credentials and making simple API calls.

The other folders are code from deeplearning.ai that has been updated to work with OpenAI version 1.57. The code examples are run outside the jupyter notebook.

## Completion

The completion.py makes a direct API call to OpenAI. It shows how to view the structure of the returned response. This is useful when you want to navigate the response object and extract the answer.

- cust.py uses prompt template to make direct OpenAI call
- hichain.py uses prompt template with langchain API
- hi-chain.py uses a prompt with langchain API
- api.py lists all the OpenAI models. If you can run this, OpenAI APi credentials are setup correctly.
- version.py prints the version of python used
- style.py upgrades the code example in deeplearning.ai to use the latest version

## Install Dependencies

Run:

```
pip install -r requirements.txt
```


Here's the code updated to the latest version of langchain:

```python
from langchain_openai import ChatOpenAI

# Initialize the chat model
chat = ChatOpenAI(temperature=0.0, model=llm_model)

# Define the template string
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

# Create prompt template
from langchain_core.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(template_string)

# The following lines remain the same since they're just checking template properties
prompt_template.messages[0].prompt
prompt_template.messages[0].prompt.input_variables

# Define style and text (these remain unchanged)
customer_style = """American English \
in a calm and respectful tone
"""

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

# Format messages (remains the same)
customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)

# Type checking (remains the same)
print(type(customer_messages))
print(type(customer_messages[0]))

# Print message (remains the same)
print(customer_messages[0])
```

The main changes are:
1. `from langchain.chat_models import ChatOpenAI` → `from langchain_openai import ChatOpenAI`
2. `from langchain.prompts import ChatPromptTemplate` → `from langchain_core.prompts import ChatPromptTemplate`

Everything else remains functionally the same. Make sure you have the required packages installed:
```bash
pip install langchain-openai langchain-core
```
