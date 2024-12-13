from dotenv import load_dotenv

import os
import sys
import openai

load_dotenv()  # This will automatically look for .env in the same directory
API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = API_KEY