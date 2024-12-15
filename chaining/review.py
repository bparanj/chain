from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path

import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creds import openai

# pip install pandas

def create_review_processing_chain():
    # Initialize the chat model
    llm = ChatOpenAI(
        temperature=0.9,
        model="gpt-3.5-turbo"
    )

    # Chain 1: Translate to English
    translate_prompt = ChatPromptTemplate.from_template(
        "Translate the following review to English:\n\n{Review}"
    )
    translate_chain = translate_prompt | llm | StrOutputParser()

    # Chain 2: Summarize
    summarize_prompt = ChatPromptTemplate.from_template(
        "Can you summarize the following review in 1 sentence:\n\n{Review}"
    )
    summarize_chain = summarize_prompt | llm | StrOutputParser()

    # Chain 3: Detect Language
    language_prompt = ChatPromptTemplate.from_template(
        "What language is the following review:\n\n{Review}"
    )
    language_chain = language_prompt | llm | StrOutputParser()

    # Chain 4: Generate Follow-up
    followup_prompt = ChatPromptTemplate.from_template(
        """Write a follow-up response to the following summary in the specified language:
        Summary: {summary}
        Language: {language}"""
    )
    followup_chain = followup_prompt | llm | StrOutputParser()

    # Create the combined chain
    def process_review(inputs):
        # Get English translation
        english_review = translate_chain.invoke({"Review": inputs["Review"]})

        # Get summary (using English review)
        summary = summarize_chain.invoke({"Review": english_review})

        # Detect language of original review
        language = language_chain.invoke({"Review": inputs["Review"]})

        # Generate follow-up
        followup = followup_chain.invoke({
            "summary": summary,
            "language": language
        })

        # Return all results
        return {
            "English_Review": english_review,
            "summary": summary,
            "language": language,
            "followup_message": followup
        }

    return process_review

def main():
    try:
        # Load your DataFrame (replace with your actual data loading logic)
        df = pd.read_csv(Path(__file__).resolve().parent / 'data.csv')

        # Create the processing chain
        process_review = create_review_processing_chain()

        # Get a review (replace 5 with your desired index)
        review = df.Review[5]
        print(f"\nProcessing review: {review[:100]}...")  # Show first 100 chars

        # Process the review
        response = process_review({"Review": review})

        # Print results
        print("\nResults:")
        print("-" * 50)
        print("English Review:", response["English_Review"])
        print("\nSummary:", response["summary"])
        print("\nDetected Language:", response["language"])
        print("\nFollow-up Message:", response["followup_message"])

    except FileNotFoundError:
        print("Error: Please ensure your data file exists and update the file path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()