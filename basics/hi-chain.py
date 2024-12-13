from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
  SystemMessage("Translate the following from English to French"),
  HumanMessage("hi!")
]

response = model.invoke(messages)

print(response.content)