
import chromadb
from chromadb.utils import embedding_functions
from review_data_etl import prepare_reviews_data
from chroma_utils import build_chroma_collection

DATA_PATH = "data/*"
CHROMA_PATH = "review_embeddings"
EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1" 
COLLECTION_NAME = "app_reviews"


# Running query
client = chromadb.PersistentClient(CHROMA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_FUNC_NAME)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

# Good reviews
great_reviews = collection.query(query_texts=["Find me some positive reviews that discuss the apps performance"],
                                 n_results=5,
                                 include=["documents", "distances", "metadatas"])

great_reviews["documents"][0][0]

# Bad reviews
bad_reviews = collection.query(query_texts=["Find me reviews of people who did not like the app"],
                                 n_results=5,
                                 include=["documents", "distances", "metadatas"])
bad_reviews["documents"][0][0]