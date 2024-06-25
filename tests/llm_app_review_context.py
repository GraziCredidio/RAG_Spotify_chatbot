import os
import json
import openai
import chromadb
from chromadb.utils import embedding_functions

# Set upt environment variable (suppress a warning related to huggingface tokenizers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Config variables
DATA_PATH = "data/*"
CHROMA_PATH = "review_embeddings"
EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1" # model specifically trained to solve question-and-answer semantic search tasks
COLLECTION_NAME = "app_reviews"

# Loading openAI API key
with open("config.json", mode="r") as json_file:
    config_data = json.load(json_file)

openai.api_key = config_data.get("openai-secret-key")

# Initialize chromadb client
client = chromadb.PersistentClient(CHROMA_PATH)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
     model_name=EMBEDDING_FUNC_NAME
     )

collection = client.get_collection(
     name=COLLECTION_NAME, embedding_function=embedding_func
     )


# Defining the context and questions to the LLM
context = """ 
You are a customer success employee at a large
audio streaming and media service provider company. 
Use the following app reviews to answer questions: {}
 """

# More generic question about good reviews
question_good_reviews = """ 
What's the key to great customer satisfaction
based on detailed positive reviews?
"""

good_reviews = collection.query(
    query_texts=[question_good_reviews],
    n_results=10,
    include=["documents"],
    where={"Rating": {"$gte": 3}}, # filter for ratings greater than 3
)

good_reviews_str = ",".join(good_reviews["documents"][0])

good_review_summaries = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": context.format(good_reviews_str)}, # providing reviews as context
        {"role": "user", "content": question_good_reviews},
        ],
    temperature=0,
    n=1,
)

print(good_review_summaries["choices"][0]["message"]["content"])

# More specific question about bad reviews
question_bad_reviews = """ 
Among poor reviews, which one has the
worst implications about the app? 
Explain why.
"""

bad_reviews = collection.query(
    query_texts=[question_bad_reviews],
    n_results=10,
    include=["documents"],
    where={"$and": [{"Rating": {"$gte": 0}}, {"Rating":{"$lte": 3}}]}, # filter for ratings less than 3 (between 0 and 3)
)

bad_reviews_str = ",".join(bad_reviews["documents"][0])

bad_review_summaries = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": context.format(bad_reviews_str)}, # providing reviews as context
        {"role": "user", "content": question_bad_reviews},
        ],
    temperature=0,
    n=1,
)

print(bad_review_summaries["choices"][0]["message"]["content"])