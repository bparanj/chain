from langchain_openai import ChatOpenAI

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creds import openai

# Initialize the chat model
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
service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""

service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""

# Format messages
service_messages = prompt_template.format_messages(
    style=service_style_pirate,
    text=service_reply)

# Type checking
print(type(service_messages))
print(service_messages[0].content)

# Print message
#print(customer_messages[0])

response = chat.invoke(service_messages)
print(response.content)
