The basics folder consists of setup instructions and programs to check the OpenAI credentials and making simple API calls.

The other folders are code from deeplearning.ai that has been updated to work with OpenAI version 1.57. The code examples are run outside the jupyter notebook.

## Completion

The completion.py makes a direct API call to OpenAI. It shows how to view the structure of the returned response. This is useful when you want to navigate the response object and extract the answer.

- cust.py uses prompt template to make direct OpenAI call
- hichain.py uses prompt template with langchain API
- hi-chain.py uses a prompt with langchain API
- api.py lists all the OpenAI models. If you can run this, OpenAI APi credentials are setup correctly.
- version.py prints the version of python used
- style.py upgrades the code example in deeplearning.ai to use the latest version

## Install Dependencies

Run:

```
pip install -r requirements.txt
```
