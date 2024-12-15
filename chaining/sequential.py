from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creds import openai

def create_sequential_chain():
    # Initialize the chat model
    llm = ChatOpenAI(
        temperature=0.9,
        model="gpt-3.5-turbo"
    )
    
    # Create first prompt for company name
    first_prompt = ChatPromptTemplate.from_template(
        "What is the best name to describe a company that makes {product}?"
    )
    first_chain = first_prompt | llm
    
    # Create second prompt for company description
    second_prompt = ChatPromptTemplate.from_template(
        "Write a 20-word description for the following company: {text}"
    )
    second_chain = second_prompt | llm
    
    # Combine the chains using the new pipe syntax
    overall_chain = first_chain | second_chain
    
    return overall_chain

def main():
    # Create the sequential chain
    chain = create_sequential_chain()
    
    # Define the product
    product = "Queen Size Sheet Set"
    
    print(f"\nGenerating company name and description for: {product}")
    print("-" * 50)
    
    # Generate response
    response = chain.invoke({"product": product})
    
    # Print the final result
    print("\nFinal Description:", response.content)
    
if __name__ == "__main__":
    main()