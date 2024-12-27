from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict, Any
from collections import deque

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creds import openai

class WindowedChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, k: int = 1):
        """Initialize with a window size of k message pairs"""
        self.messages: deque = deque(maxlen=k * 2)  # *2 because each interaction has 2 messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the window, automatically removing older ones if needed"""
        self.messages.append(message)

    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()

    def get_messages(self) -> List[BaseMessage]:
        """Get all messages currently in the window"""
        return list(self.messages)

def create_chat_chain():
    """Create a chat chain with the specified prompt template"""
    llm = ChatOpenAI(temperature=0.0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. You only remember the last interaction."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = prompt | llm
    return chain

def demonstrate_window_memory():
    """Demonstrate basic window memory operations"""
    history = WindowedChatMessageHistory(k=1)

    print("\nDemonstrating window memory operations...")

    # Save first context
    history.add_message(HumanMessage(content="Hi"))
    history.add_message(AIMessage(content="What's up"))

    print("\nAfter first interaction:")
    for msg in history.get_messages():
        print(f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}")

    # Save second context (should push out the first one)
    history.add_message(HumanMessage(content="Not much, just hanging"))
    history.add_message(AIMessage(content="Cool"))

    print("\nAfter second interaction (window size=1):")
    for msg in history.get_messages():
        print(f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}")

def chat_with_window_memory():
    """Run a conversation using windowed memory"""
    chain = create_chat_chain()
    history = WindowedChatMessageHistory(k=1)

    def process_message(input_text: str) -> str:
        # Generate response using current history window
        response = chain.invoke({
            "chat_history": history.get_messages(),
            "input": input_text
        })

        # Save the new interaction to history
        history.add_message(HumanMessage(content=input_text))
        history.add_message(AIMessage(content=response.content))

        return response.content

    # Run the conversation
    print("\nStarting conversation...")
    inputs = [
        "Hi, my name is Andrew",
        "What is 1+1?",
        "What is my name?"
    ]

    responses = []
    for user_input in inputs:
        print(f"\nHuman: {user_input}")
        response = process_message(user_input)
        print(f"Assistant: {response}")
        responses.append(response)

        print("\nCurrent memory window:")
        for msg in history.get_messages():
            print(f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}")

    return responses

if __name__ == "__main__":
    # First demonstrate basic window memory operations
    demonstrate_window_memory()

    # Then run the conversation with window memory
    print("\n" + "="*50 + "\n")
    responses = chat_with_window_memory()