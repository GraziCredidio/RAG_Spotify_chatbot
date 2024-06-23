import chromadb
from chromadb.utils import embedding_functions
from review_data_etl import prepare_reviews_data
from chroma_utils import build_chroma_collection


DATA_PATH = "data/*"
CHROMA_PATH = "review_embeddings"
EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1" # model specifically trained to solve question-and-answer semantic search tasks
COLLECTION_NAME = "app_reviews"

chroma_reviews_dict = prepare_reviews_data(DATA_PATH)

build_chroma_collection(
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_FUNC_NAME,
    chroma_reviews_dict["ids"],
    chroma_reviews_dict["documents"],
    chroma_reviews_dict["metadatas"]
)