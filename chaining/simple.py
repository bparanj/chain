from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creds import openai

def create_naming_chain():
    # Initialize the chat model
    llm = ChatOpenAI(
        temperature=0.9,
        model="gpt-3.5-turbo"
    )
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(
        "What is the best name to describe a company that makes {product}?"
    )
    
    # Create the chain using the new pipe syntax
    chain = prompt | llm
    
    return chain

def main():
    # Create the chain
    chain = create_naming_chain()
    
    # Define the product
    product = "Queen Size Sheet Set"
    
    # Generate response using invoke
    response = chain.invoke({"product": product})
    
    # Print the response
    print("\nProduct:", product)
    print("Suggested company name:", response.content)

if __name__ == "__main__":
    main()