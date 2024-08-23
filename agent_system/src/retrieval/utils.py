import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)


# Tools for retrieval ===================
def get_top_k_similar(query_embeddings, document_embeddings, K):
    """
    Get the indices of the top k most similar document embeddings for each query embedding.

    Args:
        query_embeddings (np.array): An array of query embeddings.
        document_embeddings (np.array): An array of document embeddings.
        K (int): The number of most similar document embeddings to retrieve.

    Returns:
        np.array: An array of indices of the top k most similar document embeddings for each query.
    """
    # Normalize the embeddings to unit vectors
    query_norm = query_embeddings / np.linalg.norm(
        query_embeddings, axis=1, keepdims=True
    )
    document_norm = document_embeddings / np.linalg.norm(
        document_embeddings, axis=1, keepdims=True
    )

    # Compute the cosine similarity
    similarity_matrix = np.dot(query_norm, document_norm.T)

    # Get the indices of the top k closest document embeddings for each query
    top_k_indices = np.argsort(-similarity_matrix, axis=1)
    if K is not None:
        top_k_indices = top_k_indices[:, :K]

    return top_k_indices


def mean_reciprocal_rank(df):
    """
    Compute the mean reciprocal rank for a dataframe of query results.
    df should have columns 'qid' and 'rank'.

    Args:
        df (pd.DataFrame): A dataframe containing query results with columns 'qid', 'rank'.

    Returns:
        float: The mean reciprocal rank.
    """
    # Group by query_id
    grouped = df.groupby("qid")
    # Compute the reciprocal rank for each query
    reciprocal_ranks = grouped.apply(
        lambda x: 1 / x["rank"].min(), include_groups=False
    )
    return reciprocal_ranks.mean()


def precision_recall_at_k(df, K):
    """
    Compute the mean precision and recall at K for a dataframe of query results.
    df should have columns 'qid', 'rank' columns

    Args:
        df (pd.DataFrame): A dataframe containing query results with columns 'qid', 'rank'.
        K (int): The rank cutoff for computing precision and recall.

    Returns:
        tuple: A tuple containing the mean precision and recall at K.
    """
    # Group by query_id
    grouped = df.groupby("qid")

    precision_list = []
    recall_list = []
    for query_id, group in grouped:
        # Get top K documents
        top_k = group.nsmallest(K, "rank")

        # Number of relevant documents in the top K
        num_relevant_in_top_k = top_k["rank"].le(K).sum()  # le(K) checks if rank <= K

        # Total number of relevant documents for this query
        total_relevant = len(group[group["rank"] <= K])

        # Precision@K
        precision = num_relevant_in_top_k / K

        # Recall@K
        recall = num_relevant_in_top_k / total_relevant if total_relevant > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

    # Mean precision and recall
    mean_precision = sum(precision_list) / len(precision_list)
    mean_recall = sum(recall_list) / len(recall_list)

    return mean_precision, mean_recall
