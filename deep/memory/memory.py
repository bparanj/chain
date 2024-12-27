from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict, Any, List
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creds import openai

class SimpleChatMessageHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages: List[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []

    def get_messages(self) -> List[BaseMessage]:
        return self.messages

def create_chat_chain():
    # Initialize the model
    llm = ChatOpenAI(temperature=0.0)

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Create the chain
    chain = prompt | llm

    return chain

def chat_with_memory():
    # Initialize the components
    chain = create_chat_chain()
    history = SimpleChatMessageHistory()

    def process_message(input_text: str) -> str:
        # Get current chat history
        messages = history.get_messages()

        # Generate response
        response = chain.invoke({
            "chat_history": messages,
            "input": input_text
        })

        # Save the interaction to history
        history.add_message(HumanMessage(content=input_text))
        history.add_message(AIMessage(content=response.content))

        return response.content

    # Run sample conversation
    print("\nStarting conversation...")

    responses = []
    inputs = [
        "Hi, my name is Andrew",
        "What is 1+1?",
        "What is my name?"
    ]

    for user_input in inputs:
        print(f"\nHuman: {user_input}")
        response = process_message(user_input)
        print(f"Assistant: {response}")
        responses.append(response)

    # Print full conversation history
    print("\nFull Chat History:")
    for msg in history.get_messages():
        role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
        print(f"{role}: {msg.content}")

    return responses, history

# Run the main conversation
responses, history = chat_with_memory()

# Demonstrate basic history operations
def demonstrate_history_operations():
    print("\nDemonstrating separate history operations...")
    new_history = SimpleChatMessageHistory()

    # Add some messages
    new_history.add_message(HumanMessage(content="Hi"))
    new_history.add_message(AIMessage(content="What's up"))

    print("\nInitial messages:")
    for msg in new_history.get_messages():
        print(f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}")

    # Add more messages
    new_history.add_message(HumanMessage(content="Not much, just hanging"))
    new_history.add_message(AIMessage(content="Cool"))

    print("\nFinal messages:")
    for msg in new_history.get_messages():
        print(f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}")

if __name__ == "__main__":
    demonstrate_history_operations()

