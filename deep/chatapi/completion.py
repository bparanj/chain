import os
import json
import sys
from openai import OpenAI

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

