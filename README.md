The basics folder consists of setup instructions and programs to check the OpenAI credentials and making simple API calls.

The other folders are code from deeplearning.ai that has been updated to work with OpenAI version 1.57. The code examples are run outside the jupyter notebook.

## Completion

The completion.py makes a direct API call to OpenAI. It shows how to view the structure of the returned response. This is useful when you want to navigate the response object and extract the answer.

- cust.py uses prompt template to make direct OpenAI call
- hichain.py uses prompt template with langchain API
- hi-chain.py uses a prompt with langchain API
- api.py lists all the OpenAI models. If you can run this, OpenAI APi credentials are setup correctly.
- version.py prints the version of python used

## Install Dependencies

Run:

```
pip install -r requirements.txt
```

## Test

Below is an updated example compatible with `langchain==0.3.11` and `openai==1.57.2`.  
Replace `"YOUR_API_KEY"` with your actual API key.

```python
import openai
from langchain.llms import OpenAI

openai.api_key = "YOUR_API_KEY"

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English \
in a calm and respectful tone
"""

prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

# Create an LLM instance from LangChain
llm = OpenAI(temperature=0)

# Generate a completion
response = llm(prompt)

print(response)
```
