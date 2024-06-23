import pathlib
import polars as pl

def prepare_reviews_data(data_path: pathlib.Path):
    """" Prepare the Spotify app reviews dataset for ChromaDB"""
    
    #Define schema
    dtypes = {
        "Time_submitted": pl.Datetime,
        "Review": pl.Utf8,
        "Rating": pl.Float64,
        "Total_thumbsup": pl.Float64,
        "Reply": pl.Utf8
    }
    
    # Scan the Spotify app reviews dataset
    app_reviews = pl.scan_csv(data_path, dtypes = dtypes)
    
    # Extract time submitted as date and time new columns
    app_review_db_data = (
        app_reviews.with_columns(
            [
                (
                    pl.col("Time_submitted").str.split(
                        by = " ").list.get(0).cast(pl.Date)
                ).alias("Review_date"),
                (pl.col("Time_submitted").str.split(by = " ").list.get(1)).alias(
                    "Review_time").cast(pl.Time),
            ]   
        )
        .select(["Review", "Rating", "Total_thumbsup", "Review_date", "Review_time", "Reply"])
        .sort(["Review_date", "Rating"])
        .collect()    
    )
    
    # Create ids, documents, metadatas in the chromadb format
    ids = [f"review{i}" for i in range(app_review_db_data.shape[0])]
    documents = app_review_db_data["Review"].to_list()
    metadatas = app_review_db_data.drop("Review").to_dicts()
    
    return{"ids": ids, "documents": documents, "metadatas": metadatas}