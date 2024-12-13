from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from pathlib import Path

# Get the project root directory
root_dir = Path(__file__).resolve().parent.parent
dotenv_path = os.path.join(root_dir, '.env')

# Load the .env file
load_dotenv(dotenv_path)

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI()

def get_completion(prompt , model="text-davinci-003"):
  response = client.chat.completions.create(
    messages=[
      {
        "role": "user",
        "content": prompt
      }
    ],
    model="gpt-4o"
  )

  return response

result = get_completion("Bugs Bunny is ")

print("\n" + "="*50 + "\n")
print(type(result))
print("\n" + "="*50 + "\n")

from pprint import pprint
pprint(vars(result), indent=2, width=80)

print("\n" + "="*50 + "\n")

answer = result.choices[0].message.content

print(answer)
