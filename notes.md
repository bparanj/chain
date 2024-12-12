pyenv: pip: command not found

The `pip' command exists in these Python versions:
  3.10.16

Note: See 'pyenv help global' for tips on allowing both
      python2 and python3 to be found.

This error occurs when pip isn't available in your current Python environment. Since pyenv shows that pip exists in Python 3.10.16, here are a few steps to resolve this:

1. Either set Python 3.10.16 as your global Python version:
```bash
pyenv global 3.10.16
```

2. Or activate Python 3.10.16 for your current shell session:
```bash
pyenv shell 3.10.16
```

After doing either of these, pip should be available. You can verify by running:
```bash
pip --version
```

If you still have issues, you might need to rehash pyenv to update its shims:
```bash
pyenv rehash
```

  raise self._make_status_error_from_response(err.response) from None
openai.RateLimitError: Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}

Buy credits

Ah, you need to install the `python-dotenv` package first. Run this command:

```bash
pip install python-dotenv
```

If that doesn't work (due to pip issues with pyenv that you mentioned earlier), you can try:
```bash
python -m pip install python-dotenv
```

Or if you want a simpler version of the program without using dotenv, here's an alternative:

```python
import os
from openai import OpenAI

# Initialize the client with API key directly from environment variable
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# List all models
try:
    models = client.models.list()

    print("\nAvailable OpenAI Models:")
    print("-" * 50)
    for model in models:
        print(f"ID: {model.id}")
        print(f"Created: {model.created}")
        print(f"Owner: {model.owned_by}")
        print("-" * 50)

except Exception as e:
    print(f"An error occurred: {str(e)}")
```

Just make sure your OpenAI API key is set as an environment variable before running the script.

We can convert the Unix timestamp to a readable date using Python's `datetime` module. Here's the modified code:

```python
import os
from openai import OpenAI
from datetime import datetime

# Initialize the client with API key directly from environment variable
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# List all models
try:
    models = client.models.list()

    print("\nAvailable OpenAI Models:")
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
```

You can customize the date format by modifying the `strftime` format string. Some common formats:
- `%Y-%m-%d`: 2024-03-21
- `%B %d, %Y`: March 21, 2024
- `%d/%m/%Y`: 21/03/2024
- `%Y-%m-%d %H:%M:%S`: 2024-03-21 14:30:45

https://platform.openai.com/docs/api-reference/models

Available OpenAI Models:
Total models: 42
--------------------------------------------------
ID: text-embedding-3-small
Created: 2024-01-22 13:43:17
Owner: system
--------------------------------------------------
ID: gpt-4o-realtime-preview
Created: 2024-09-29 21:33:18
Owner: system
--------------------------------------------------
ID: dall-e-2
Created: 2023-10-31 20:22:57
Owner: system
--------------------------------------------------
ID: gpt-4o-realtime-preview-2024-10-01
Created: 2024-09-23 18:49:26
Owner: system
--------------------------------------------------
ID: o1-mini
Created: 2024-09-06 14:56:48
Owner: system
--------------------------------------------------
ID: o1-mini-2024-09-12
Created: 2024-09-06 14:56:19
Owner: system
--------------------------------------------------
ID: gpt-4-1106-preview
Created: 2023-11-02 16:33:26
Owner: system
--------------------------------------------------
ID: babbage-002
Created: 2023-08-21 12:16:55
Owner: system
--------------------------------------------------
ID: davinci-002
Created: 2023-08-21 12:11:41
Owner: system
--------------------------------------------------
ID: whisper-1
Created: 2023-02-27 16:13:04
Owner: openai-internal
--------------------------------------------------
ID: dall-e-3
Created: 2023-10-31 16:46:29
Owner: system
--------------------------------------------------
ID: text-embedding-ada-002
Created: 2022-12-16 14:01:39
Owner: openai-internal
--------------------------------------------------
ID: gpt-4o-2024-08-06
Created: 2024-08-04 19:38:39
Owner: system
--------------------------------------------------
ID: gpt-3.5-turbo-16k
Created: 2023-05-10 18:35:02
Owner: openai-internal
--------------------------------------------------
ID: gpt-4-0125-preview
Created: 2024-01-23 14:20:12
Owner: system
--------------------------------------------------
ID: gpt-4o
Created: 2024-05-10 14:50:49
Owner: system
--------------------------------------------------
ID: o1-preview-2024-09-12
Created: 2024-09-06 14:54:25
Owner: system
--------------------------------------------------
ID: gpt-4-turbo-preview
Created: 2024-01-23 14:22:57
Owner: system
--------------------------------------------------
ID: omni-moderation-latest
Created: 2024-11-15 11:47:45
Owner: system
--------------------------------------------------
ID: gpt-4o-2024-05-13
Created: 2024-05-10 15:08:52
Owner: system
--------------------------------------------------
ID: omni-moderation-2024-09-26
Created: 2024-11-27 14:07:46
Owner: system
--------------------------------------------------
ID: tts-1-hd-1106
Created: 2023-11-03 19:18:53
Owner: system
--------------------------------------------------
ID: gpt-4o-mini
Created: 2024-07-16 19:32:21
Owner: system
--------------------------------------------------
ID: chatgpt-4o-latest
Created: 2024-08-12 22:12:11
Owner: system
--------------------------------------------------
ID: gpt-4
Created: 2023-06-27 12:13:31
Owner: openai
--------------------------------------------------
ID: gpt-4-0613
Created: 2023-06-12 12:54:56
Owner: openai
--------------------------------------------------
ID: o1-preview
Created: 2024-09-06 14:54:57
Owner: system
--------------------------------------------------
ID: gpt-4o-mini-2024-07-18
Created: 2024-07-16 19:31:57
Owner: system
--------------------------------------------------
ID: gpt-3.5-turbo
Created: 2023-02-28 13:56:42
Owner: openai
--------------------------------------------------
ID: gpt-3.5-turbo-0125
Created: 2024-01-23 17:19:18
Owner: system
--------------------------------------------------
ID: gpt-4o-2024-11-20
Created: 2024-11-18 19:10:40
Owner: system
--------------------------------------------------
ID: gpt-4-turbo
Created: 2024-04-05 19:57:21
Owner: system
--------------------------------------------------
ID: tts-1-hd
Created: 2023-11-03 17:13:35
Owner: system
--------------------------------------------------
ID: gpt-4-turbo-2024-04-09
Created: 2024-04-08 14:41:17
Owner: system
--------------------------------------------------
ID: gpt-3.5-turbo-1106
Created: 2023-11-02 17:15:48
Owner: system
--------------------------------------------------
ID: gpt-4o-audio-preview
Created: 2024-09-27 14:07:23
Owner: system
--------------------------------------------------
ID: gpt-4o-audio-preview-2024-10-01
Created: 2024-09-26 18:17:22
Owner: system
--------------------------------------------------
ID: gpt-3.5-turbo-instruct
Created: 2023-08-24 14:23:47
Owner: system
--------------------------------------------------
ID: tts-1
Created: 2023-04-19 17:49:11
Owner: openai-internal
--------------------------------------------------
ID: tts-1-1106
Created: 2023-11-03 19:14:01
Owner: system
--------------------------------------------------
ID: gpt-3.5-turbo-instruct-0914
Created: 2023-09-07 17:34:32
Owner: system
--------------------------------------------------
ID: text-embedding-3-large
Created: 2024-01-22 14:53:00
Owner: system
--------------------------------------------------
