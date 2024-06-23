import pathlib
import polars as pl

def prepare_reviews_data(data_path: pathlib.Path):
    """" Prepare the Spotify app reviews dataset for ChromaDB"""
    
    #Define schema
    dtypes = {
        "Time_submitted": pl.Utf8,
        "Review": pl.Utf8,
        "Rating": pl.Float64,
        "Total_thumbsup": pl.Float64,
        "Reply": pl.Utf8
    }
    
    # Scan the Spotify app reviews dataset
    app_reviews = pl.scan_csv(data_path, dtypes = dtypes)
    
    # Extract time submitted as date and time new columns
    app_review_db_data = (
        app_reviews.with_columns()
        .select(["Time_submitted","Review", "Rating", "Total_thumbsup", "Reply"])
        .sort(["Time_submitted", "Rating"])
        .collect()    
    )
    
    # Create ids, documents, metadatas in the chromadb format
    ids = [f"review{i}" for i in range(app_review_db_data.shape[0])]
    documents = app_review_db_data["Review"].to_list()
    metadatas = app_review_db_data.drop("Review").to_dicts()
    
    return {"ids": ids, "documents": documents, "metadatas": metadatas}