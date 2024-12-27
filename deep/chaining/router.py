from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creds import openai

# Define templates
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""

math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts,\
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""

computerscience_template = """You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity.

Here is a question:
{input}"""

# Define prompt information
prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template
    },
    {
        "name": "history",
        "description": "Good for answering history questions",
        "prompt_template": history_template
    },
    {
        "name": "computer science",
        "description": "Good for answering computer science questions",
        "prompt_template": computerscience_template
    }
]

# Initialize the LLM
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"
)

# Create destination chains
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate.from_template(template=prompt_template)
    chain = prompt | llm
    destination_chains[name] = chain

# Create destinations string
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# Create default chain
default_prompt = PromptTemplate.from_template("{input}")
default_chain = default_prompt | llm

# Define router template with proper escaping
MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a JSON object formatted to look like:
{{
    "destination": string,  // name of the prompt to use or "DEFAULT"
    "next_inputs": string  // a potentially modified version of the original input
}}

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{input}

<< OUTPUT >>"""

# Create router prompt
router_prompt = PromptTemplate(
    template=MULTI_PROMPT_ROUTER_TEMPLATE,
    input_variables=["input", "destinations"],
)

# Create output parser
class RouterOutputParser(JsonOutputParser):
    def parse(self, text: str) -> dict:
        cleaned_text = text.replace("```json", "").replace("```", "")
        return super().parse(cleaned_text)

# Create the router chain
router_chain = router_prompt | llm | RouterOutputParser()

# Function to route to the appropriate chain
def route_to_chain(inputs):
    try:
        router_result = router_chain.invoke({
            "input": inputs["input"],
            "destinations": destinations_str
        })

        destination = router_result["destination"].lower()
        next_inputs = {"input": router_result["next_inputs"]}

        # If destination exists in destination_chains, use it; otherwise use default
        if destination in destination_chains:
            return destination_chains[destination].invoke(next_inputs)
        else:
            print(f"\nUsing default chain for input: {next_inputs['input']}")
            return default_chain.invoke(next_inputs)

    except Exception as e:
        print(f"\nEncountered an error, using default chain. Error: {str(e)}")
        return default_chain.invoke({"input": inputs["input"]})

# Create the final chain
chain = RunnablePassthrough() | RunnableLambda(route_to_chain)

# Example usage
if __name__ == "__main__":
    questions = [
        "What is black body radiation?",
        "what is 2 + 2",
        "Why does every cell in our body contain DNA?",
        "What was the first song ever written?",  # This should go to default
        "What is the meaning of life?"  # This should go to default
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        response = chain.invoke({"input": question})
        print(f"Response: {response}")