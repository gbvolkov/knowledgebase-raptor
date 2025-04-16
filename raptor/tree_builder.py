# tree_builder.py
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from embeddings.embedder import get_embedding_model
from raptor.clustering import perform_clustering
from raptor.summarizer import summarize_text

# Load the embedding model (using intfloat/multilingual-e5-large)
embd = get_embedding_model()

def embed(texts: List[str]) -> np.ndarray:
    """
    Generates embeddings for a list of text documents.
    """
    text_embeddings = embd.embed_documents(texts)
    return np.array(text_embeddings)

def embed_cluster_texts(texts: List[str]) -> pd.DataFrame:
    """
    Embeds texts and clusters them.
    Returns a DataFrame with the texts, their embeddings, and cluster labels.
    """
    text_embeddings_np = embed(texts)
    cluster_labels = perform_clustering(text_embeddings_np, dim=10, threshold=0.1)
    df = pd.DataFrame()
    df["text"] = texts
    df["embd"] = list(text_embeddings_np)
    df["cluster"] = cluster_labels
    return df

def fmt_txt(df: pd.DataFrame) -> str:
    """
    Formats the texts in a DataFrame by joining them with a delimiter.
    """
    unique_txt = df["text"].tolist()
    return "\n\n---\n\n".join(unique_txt)

def embed_cluster_summarize_texts(texts: List[str], level: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Embeds, clusters, and then summarizes texts.
    Returns two DataFrames: one for clusters and one for summaries.
    """
    df_clusters = embed_cluster_texts(texts)
    expanded_list = []
    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append({"text": row["text"], "embd": row["embd"], "cluster": cluster})
    expanded_df = pd.DataFrame(expanded_list)
    all_clusters = expanded_df["cluster"].unique()
    print(f"--Generated {len(all_clusters)} clusters at level {level}--")
    summaries = []
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        formatted_txt = fmt_txt(df_cluster)
        summaries.append(summarize_text(formatted_txt))
    df_summary = pd.DataFrame({
        "summaries": summaries,
        "level": [level] * len(summaries),
        "cluster": list(all_clusters),
    })
    return df_clusters, df_summary

def recursive_embed_cluster_summarize(texts: List[str], level: int = 1, n_levels: int = 3) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Recursively builds the RAPTOR tree by embedding, clustering, and summarizing texts.
    """
    results = {}
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)
    results[level] = (df_clusters, df_summary)
    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(new_texts, level + 1, n_levels)
        results.update(next_level_results)
    return results
