import os
from openai import OpenAI
from datetime import datetime

# Initialize the client with API key directly from environment variable
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# List all models
try:
    models = client.models.list()

    print("\nAvailable OpenAI Models:")
    print(f"Total models: {len(list(client.models.list()))}")
    print("-" * 50)
    for model in models:
        # Convert Unix timestamp to datetime
        created_date = datetime.fromtimestamp(model.created)
        # Format the date (you can adjust the format as needed)
        formatted_date = created_date.strftime("%Y-%m-%d %H:%M:%S")

        print(f"ID: {model.id}")
        print(f"Created: {formatted_date}")
        print(f"Owner: {model.owned_by}")
        print("-" * 50)

except Exception as e:
    print(f"An error occurred: {str(e)}")