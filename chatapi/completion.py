from openai import OpenAI
import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from creds import API_KEY

client = OpenAI(api_key=API_KEY)

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
