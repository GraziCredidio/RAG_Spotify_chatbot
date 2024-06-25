# RAG Spotify chatbot

## Overview
The deliverable of this project is an internal chatbot, which stakeholders can use to ask questions about the company's data (e.g.: Spotify app reviews) through an easy-to-distribute streamlit application. It can be used to summarize and provide insights to address poor user experience, for example. This way, stakeholders can have answers to ad-hoc questions without having to use query languages, request reports from analysts or wait for dashboards to be created to actionable and direct questions. 

To make LLM responses more tailored to the business (e.g.: Spotify), some reviews are provided as context to the LLM. The chatbot accepts a query, finds semantically similar documents and uses them as context to the LLM. This **Retrieval-augmented generation (RAG)** allows the model to use information that was not originally in its training dataset. 

A vector database was used to store embedded reviews and retrieve similar documents as described in the query. The retrieved relevant documents and the query are passed to the LLM. A context-informed response is generated as a result.  

## Technical stack 
`polars` was used in the preprocessing of data and creation of documents. A collection in bacthes of 166 documents was created and sequentially embedded and stored in a `chromaDB` vector database. The embedding model used was the `multi-qa-MiniLM-L6-cos-v1`, and `gpt-3.5-turbo` was applied as the LLM. The final deliverable web application was created using `streamlit` (with the default port 8501).  

## Files and folders
Inside `data`, you can find the [original dataset](https://www.kaggle.com/datasets/mfaaris/spotify-app-reviews-2022/data). Inside `tests`, you will find the files `test_openai.py`, `test_collection_query.py` and `llm_app_review_context.py`, that can be used to test if your OpenAI API key, collection creation and LLM + context are working as expected outside the web app. 
- `review_data_etl.py`: defines a function that will be used to prepare the app reviews dataset for ChromaDB
- `chroma_utils.py`: defines a function that will create a collection in a modular way
- `collection_creation.py`: creates collection
- `example.config.json`: file that your OpenAI secret key has to be inserted
- `app.py`: streamlit web app with the final LLM model with the Spotify app reviews as context

## Installation procedure
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/GraziCredidio/RAG_Spotify_chatbot.git
   ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Execute `collection_creation.py` to create the collection of embedded documents that will be later used as context to the LLM

4. Open the file `example.config.json` and enter your OpenAI secret key. Then, rename the file to `config.json`

5. To run the RAG Spotify chatbot web application, open the terminal in the folder and type:
    ```bash
    streamlit run app.py
    ```

