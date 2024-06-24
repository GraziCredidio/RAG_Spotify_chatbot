# RAG Spotify chatbot

## Overview
The deliverable of this project is a chatbot where stakeholders can use to ask questions about the Spotify app reviews through a easy-to-distribute streamlit application.
It can be used to summarize different types of reviews and provide insights to address poor user experience, for example. This way, stakeholders can have answers to ad-hoc questions without having to use query languages, request reports from analysts or wait for dashboards to be created to actionable and direct questions. 

To make LLM responses more tailored to the business (e.g.: Spotify), some reviews are provided as context to the LLM. The chatbot accepts a query, finds semantically similar documents and uses them as context to the LLM. This **Retrieval-augmented generation (RAG)** allows the model to use information that was not originally in its training dataset. 

A vector database was used to store embedded reviews and retrieve similar documents as described in the query. The retrieved relevant documents and the query are passed to the LLM. A context-informed response is generated as a result.  

## Technical stack 
`polars` was used in the preprocessing of data and creation of documents. A collection in bacthes of 166 documents was created and sequentially embedded and stored in a `chromaDB` vector database. The embedding model used was the `multi-qa-MiniLM-L6-cos-v1`, and `gpt-3.5-turbo` was applied as the LLM. The final deliverable web application was created using `streamlit` (with the default port 8501).  

## Installation procedure
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/GraziCredidio/RAG_Spotify_chatbot.git
   ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Execute `collection_creation.py` to create collection of documents that will be later used as context to the LLM

4. Run the RAG Spotify chatbot web application:
    ```bash
    streamlit run app.py
    ```

