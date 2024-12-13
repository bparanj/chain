Ubuntu 24 uses Python 3.12.x. Need Python 3.10.x to work with langchain.

1. Enable the Ubuntu repository that provides older Python versions. For newer Ubuntu releases, use a PPA like deadsnakes:  
   ```bash
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt update
   ```
   
2. Install Python 3.10:  
   ```bash
   sudo apt install python3.10 python3.10-venv
   ```

3. Create a virtual environment using Python 3.10:  
   ```bash
   python3.10 -m venv my_project_env
   ```

4. Activate the virtual environment:  
   ```bash
   source my_project_env/bin/activate
   ```

5. Now your project uses Python 3.10 inside the virtual environment.

Run 

```
pyenv shell 3.10.16
```

to make the project use that version.

The `version.py` is used to check the current Python version used in the project. 

Follow the langchain official tutorials to install the Python packages. The code has been modified to print the results from the OpenAI API calls.

1. Setup virtual environment
2. Hide virtual environment directory, .env and other sensitive files from Git by adding them to .gitignore
3. Run `export OPENAI_API_KEY="very-secret"`
