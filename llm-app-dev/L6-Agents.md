# LangChain: Agents

## Outline:

* Using built in LangChain tools: DuckDuckGo search and Wikipedia
* Defining your own tools


```python
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import warnings
warnings.filterwarnings("ignore")
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

## Built-in LangChain tools


```python
#!pip install -U wikipedia
```


```python
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI
```


```python
llm = ChatOpenAI(temperature=0, model=llm_model)
```


```python
tools = load_tools(["llm-math","wikipedia"], llm=llm)
```


```python
agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
```


```python
agent("What is the 25% of 300?")
```

## Wikipedia example


```python
question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"
result = agent(question) 
```

## Python Agent


```python
agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)
```


```python
customer_list = [["Harrison", "Chase"], 
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"], 
                 ["Geoff","Fusion"], 
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]
```


```python
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 
```

#### View detailed outputs of the chains


```python
import langchain
langchain.debug=True
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 
langchain.debug=False
```

## Define your own tool


```python
#!pip install DateTime
```


```python
from langchain.agents import tool
from datetime import date
```


```python
@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())
```


```python
agent= initialize_agent(
    tools + [time], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
```

**Note**: 

The agent will sometimes come to the wrong conclusion (agents are a work in progress!). 

If it does, please try running it again.


```python
try:
    result = agent("whats the date today?") 
except: 
    print("exception on external access")
```

----

from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools.python.tool import PythonREPLTool
from langchain_community.utilities.python import PythonREPL
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from datetime import date
import operator

# Initialize the chat model
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"
)

# Create tools
def create_tools():
    # Wikipedia tool
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    # Python REPL tool
    python_repl = PythonREPLTool()
    
    # Custom date tool
    def get_current_date(text: str) -> str:
        """Returns today's date. Input should be an empty string."""
        return str(date.today())
    
    date_tool = Tool(
        name="Current Date",
        func=get_current_date,
        description="Returns today's date. Use this for date-related queries."
    )
    
    # Math tool
    def calculator(expression: str) -> float:
        """Evaluates a mathematical expression."""
        try:
            return eval(expression, {"__builtins__": {}}, {
                "abs": abs, "float": float, "int": int,
                "pow": pow, "round": round, "operator": operator
            })
        except Exception as e:
            return f"Error: {str(e)}"
    
    math_tool = Tool(
        name="Calculator",
        func=calculator,
        description="Useful for mathematical calculations"
    )
    
    return [wikipedia, python_repl, date_tool, math_tool]

# Create the agent prompt
prompt_template = """You are a helpful AI assistant with access to various tools.
You should help the user with their questions and tasks using these tools when necessary.

Tools available:
{tools}

To use a tool, use the following format:
```
Thought: I need to figure out...
Action: tool_name
Action Input: input to tool
Observation: tool output
```

Begin! Remember to be helpful and thorough.
