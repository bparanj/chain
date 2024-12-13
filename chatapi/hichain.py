import os
import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from creds import API_KEY

os.environ["OPENAI_API_KEY"] = API_KEY

system_template = "Translate the following from English to {language}"
prompt_template = ChatPromptTemplate.from_messages([("system", system_template), ("user", "{text}")])

prompt = prompt_template.invoke({"language": "French", "text": "I love you"})

#print(prompt.to_messages)

model = ChatOpenAI(model="gpt-4o-mini")
response = model.invoke(prompt)

print(response.content)