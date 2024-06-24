import os
import json
import openai
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st

# Set upt environment variable (suppress a warning related to huggingface tokenizers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Config variables
DATA_PATH = "data/*"
CHROMA_PATH = "review_embeddings"
EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1"
COLLECTION_NAME = "app_reviews"

# Load OpenAI API key from config file
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

# Streamlit application
st.title("Spotify app customer reviews summarizer")

st.write("""
This application uses OpenAI's GPT-3.5 to analyze and provide insights
on the Spotify user reviews posted on Google Play Store from January to September 2022. 
""")

# User inputs
sample_context = "You are a customer success employee at a large audio streaming and media service provider company. Use the following app reviews to answer questions"
sample_question = "What's the key to great customer satisfaction based on detailed positive reviews?"

user_context = st.text_area("Enter the context for the analysis", placeholder=sample_context)
user_question = st.text_area("Enter your question", placeholder=sample_question)
rating_lower_range = st.number_input("Filter reviews with ratings above", 1, 5)
rating_higher_range = st.number_input("Filter reviews with ratings below", 1, 5, 5)

# Button to trigger analysis
if st.button("Analyze reviews"):
    with st.spinner("Analyzing reviews..."):
        reviews = collection.query(
            query_texts=[user_question],
            n_results=10,
            include=["documents"],
            where={"$and": [{ "Rating": {"$gte": rating_lower_range}}, {"Rating":{"$lte": rating_higher_range}}]}
        )
        if (rating_higher_range >= rating_lower_range): # prevent range2 to be lower than range1
            
            if reviews["documents"]: 
                reviews_str = ",".join(reviews["documents"][0])
                
                review_summaries = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": user_context.format(reviews_str)}, # providing reviews as context
                        {"role": "user", "content": user_question},
                        ],
                    temperature=0,
                    n=1,
                    )
                summary = review_summaries["choices"][0]["message"]["content"]
                
                st.subheader("Answer:")
                st.write(summary)
            else:
                st.write("No reviews found with the specified criteria.")
                
        else:
            st.write("""Rating filter is not correct. Please try again. 
                    The first range must be lower than the second range. 
                    If you do not wish to filter reviews by rating, leave the filters as: above 1 and below 5.""")
        
