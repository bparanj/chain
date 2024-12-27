from langchain_openai import ChatOpenAI

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creds import openai

# Initialize the chat model - removed the model parameter since it's not defined
chat = ChatOpenAI(temperature=0.0)

# Define the template string
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

# Create prompt template
from langchain_core.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(template_string)

# Define style and text
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

# Format messages
customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)

# Type checking
print(type(customer_messages))
print(type(customer_messages[0]))

# Print message
#print(customer_messages[0])

response = chat.invoke(customer_messages)
print(response.content)
