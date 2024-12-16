## Basics

The basics folder consists of setup instructions and programs to check the OpenAI credentials and making simple API calls.

## Folders

The other folders are code from deeplearning.ai that has been updated to work with OpenAI version 1.57 and LangChain 0.3. The code examples are run outside the jupyter notebook.

## Completion

The completion.py makes a direct API call to OpenAI. It shows how to view the structure of the returned response. This is useful when you want to navigate the response object and extract the answer.

- cust.py uses prompt template to make direct OpenAI call (Pirate to English translation - direct API call to OpenAI)
- hichain.py uses prompt template with langchain API - English to French translation using LangChain
- hi-chain.py uses a prompt with langchain API call to OpenAI for English to French translation
- api.py lists all the OpenAI models. If you can run this, OpenAI API credentials are setup correctly.
- version.py prints the version of python used by your project
- style.py - Pirate to English using LangChain

## Install Dependencies

Run:

```
pip install -r requirements.txt
```

1. Direct API call to OpenAI - What is 1+1 ? (get_completion) - cust.py
2. Direct API call to OpenAI - Translating Pirate to English (Using prompt template)
3. LangChain API call to OpenAI - Translating Pirate to English (Using prompt template)
4. LangChain API call to OpenAI - Translating English to Pirate (Using prompt template)

## Set Python Version

```
pyenv global 3.10.16
```

## Jupyter Notebook

It is easier to run the Question and Answer code examples in Jupyter notebook. It is tricky to load the given CSV file outside the notebook.

L4-QnA.ipynb uses langchain 0.3 compatible code for querying a product catalog for an item.