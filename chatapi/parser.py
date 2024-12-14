import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Define your review text
customer_review = """\
This leaf blower is pretty amazing. It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

# Define the schemas for parsing the output
gift_schema = ResponseSchema(
    name="gift",
    description="Was the item purchased as a gift? True if yes, False if not or unknown."
)
delivery_days_schema = ResponseSchema(
    name="delivery_days",
    description="How many days did it take for the product to arrive? If not found, output -1."
)
price_value_schema = ResponseSchema(
    name="price_value",
    description="Any sentences about value or price as a comma-separated Python list."
)

response_schemas = [gift_schema, delivery_days_schema, price_value_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product to arrive? \
If this information is not found, output -1.

price_value: Extract any sentences about the value or price, \
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=review_template_2)
messages = prompt.format_messages(text=customer_review, format_instructions=format_instructions)

# Create a ChatOpenAI instance with the desired model and temperature
chat = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")

# Get the response from the LLM
response = chat(messages)

# Parse the output into a dictionary
output_dict = output_parser.parse(response.content)

# output_dict now contains the structured data
print(output_dict)
print(type(output_dict))
print(output_dict.get('delivery_days'))
