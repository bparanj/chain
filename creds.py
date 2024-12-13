from dotenv import load_dotenv
import os

load_dotenv()  # This will automatically look for .env in the same directory
API_KEY = os.getenv('OPENAI_API_KEY')
