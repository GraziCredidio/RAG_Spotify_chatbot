import os
import json
import openai
os.environ["TOKENIZERS_PARALLELISM"] = "false" # suppress a warning related to huggingface tokenizers.

with open("config.json", mode="r") as json_file:
     config_data = json.load(json_file)

openai.api_key = config_data.get("openai-secret-key") # gpt-3.5-turbo LLM

# Setting up role prompt
context = "You are a customer success employee at a large a large audio streaming and media service provider company"
question = "Which features users liked the most?"

chat_response = openai.ChatCompletion.create(
     model="gpt-3.5-turbo",
     messages=[
         {"role": "system", "content": context},
         {"role": "user", "content": question},
     ],
     temperature=0,
     n=1,
)

print(chat_response["choices"][0]["message"]["content"])