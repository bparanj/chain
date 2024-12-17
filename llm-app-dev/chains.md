Notes from LangChain for LLM Application Development (deeplearning.ai)

## Chains

- Load Dot Env (read local .env file)
- Load Pandas Data Frame
- Select the Model and Set Temperature 0.9
- Chain = LLM and Prompt 
- Run Chain

### LLM Chain

- Create a LLM
- Create a Prompt Template 1
- Create Chain 1 (llm + firt prompt)
- Create Prompt Template 2
- Create Chain 2 (llm + second prompt)
- Combine the Chain 1 and Chain 2

### Sequential Chain

Useful for single input and single output.

- Prompt Template 1 (Translate to English)
- Create Chain 1 (llm + first prompt)
- Prompt Template 2 (Summarize the review)
- Create Chain 2 (llm + second prompt)
- Prompt Template 3 (Detect Language)
- Prompt Template 4 (followup message)
- Using Input and Output Keys
- Output of Chain 1 is Input to Chain 2
- Run Sequential Chain
	 Review in French
	 Review in English
	 Summary
	 Followup Message

### Router Chain

- Physics Template
- Math Template
- History Template
- CS Template
- Router Output Parser
- Create Destination Chains
- Create a Default Chain
- Define Template to Route Chains
- Create Prompt Template from Router Template
- Create Router Chain
- Create Multi Prompt Chain
- Ask Questions
	Physics
	Math
	Biology

Why do we need chains?
Why do we need sequential chains?
Why do we need router chains?
