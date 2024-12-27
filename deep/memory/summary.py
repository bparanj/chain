from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict, Any
import tiktoken

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creds import openai

class SummaryChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, llm, max_token_limit: int = 100):
        self.messages: List[BaseMessage] = []
        self.summary: str = ""
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def add_message(self, message: BaseMessage) -> None:
        """Add a message and maintain the summary if needed"""
        self.messages.append(message)

        # Check if we need to summarize
        if self._count_tokens() > self.max_token_limit:
            self._summarize_messages()

    def _count_tokens(self) -> int:
        """Count the total tokens in messages"""
        text = " ".join(msg.content for msg in self.messages)
        return len(self.encoding.encode(text))

    def _summarize_messages(self) -> None:
        """Summarize the current messages"""
        if not self.messages:
            return

        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the following conversation concisely while preserving key information:"),
            MessagesPlaceholder(variable_name="messages")
        ])

        # Generate summary
        summary_response = summary_prompt.pipe(self.llm).invoke({
            "messages": self.messages
        })

        # Store summary and clear messages except the most recent pair
        self.summary = summary_response.content
        self.messages = self.messages[-2:] if len(self.messages) >= 2 else self.messages

    def clear(self) -> None:
        """Clear all messages and summary"""
        self.messages = []
        self.summary = ""

    def get_messages(self) -> List[BaseMessage]:
        """Get current messages including summary if it exists"""
        if self.summary:
            return [SystemMessage(content=f"Previous conversation summary: {self.summary}")] + self.messages
        return self.messages

def create_chat_chain():
    """Create a chat chain with the specified prompt template"""
    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-3.5-turbo"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = prompt | llm
    return chain, llm

def main():
    # Initialize components
    chain, llm = create_chat_chain()
    history = SummaryChatMessageHistory(llm, max_token_limit=100)

    # Create schedule string
    schedule = "There is a meeting at 8am with your product team. \
    You will need your powerpoint presentation prepared. \
    9am-12pm have time to work on your LangChain \
    project which will go quickly because Langchain is such a powerful tool. \
    At Noon, lunch at the italian resturant with a customer who is driving \
    from over an hour away to meet you to understand the latest in AI. \
    Be sure to bring your laptop to show the latest LLM demo."

    # Save initial contexts
    history.add_message(HumanMessage(content="Hello"))
    history.add_message(AIMessage(content="What's up"))
    history.add_message(HumanMessage(content="Not much, just hanging"))
    history.add_message(AIMessage(content="Cool"))
    history.add_message(HumanMessage(content="What is on the schedule today?"))
    history.add_message(AIMessage(content=schedule))

    # Function to process messages
    def process_message(input_text: str) -> str:
        response = chain.invoke({
            "chat_history": history.get_messages(),
            "input": input_text
        })

        # Save the interaction
        history.add_message(HumanMessage(content=input_text))
        history.add_message(AIMessage(content=response.content))

        return response.content

    # Make prediction
    print("\nAsking about demo...")
    response = process_message("What would be a good demo to show?")
    print(f"\nResponse: {response}")

    # Show final state
    print("\nFinal conversation state:")
    for msg in history.get_messages():
        if isinstance(msg, SystemMessage):
            print(f"\nSummary: {msg.content}")
        else:
            print(f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}")

if __name__ == "__main__":
    main()
