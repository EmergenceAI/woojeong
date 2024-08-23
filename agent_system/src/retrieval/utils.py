import logging
import numpy as np
import pandas as pd

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


def evaluate_retrieval(query2apis, id2api_embed, id2query_embed, apis, queries, k_list=[5, 10]):
    """
    Evaluate the retrieval performance of a query2apis mapping using embeddings.
    
    Args:
        query2apis (dict): A mapping of query ids to lists of api ids.
        id2api_embed (dict): A mapping of api ids to api embeddings.
        id2query_embed (dict): A mapping of query ids to query embeddings.
        apis (list): A list of all api ids.
        queries (list): A list of all query ids.
        k_list (list): A list of integers specifying the ranks to compute precision and recall at.
        
    Returns:
        tuple: A tuple containing a dataframe of query-api pairs with ranks and a dictionary of evaluation metrics.
    """
    # filter mappings that are in queries
    filtered_qd_mapping = {q: apis for q, apis in query2apis.items() if q in queries}
    assert len(filtered_qd_mapping) == len(queries)

    # convert to pandas dataframe and flatten
    qd_mapping = pd.DataFrame(filtered_qd_mapping.items(), columns=['query', 'apis'])
    qd_mapping = qd_mapping.explode('apis').rename(columns={'apis': 'api'}).reset_index(drop=True)
    print(f"# of apis: {len(apis)}, # of queries: {len(queries)}, # of api calls: {len(qd_mapping)}")

    # map row idx of embedding matrix -> actual id
    idx2apiid = {i: doc_id for i, doc_id in enumerate(apis)}
    idx2qid = {i: query_id for i, query_id in enumerate(queries)}
    qid2idx = {qid: idx for idx, qid in idx2qid.items()}

    # stack embeddings
    api_embeds = np.stack([id2api_embed[id] for id in apis])
    query_embeds = np.stack([id2query_embed[id] for id in queries])
    assert len(idx2apiid) == api_embeds.shape[0]
    assert len(idx2qid) == query_embeds.shape[0]

    # compute top k documents per query
    top_k_indices = get_top_k_similar(query_embeds, api_embeds, K=None)

    # convert top_k_indices into top_k_apiids
    convert_to_id = np.vectorize(lambda x: idx2apiid[x])
    top_k_ids = convert_to_id(top_k_indices)

    # add top_k_id column to qd_mapping_filtered df
    # note that query idx should be converted to query id
    ranks = []
    for i, row in qd_mapping.iterrows():
        qid, apiid = row["query"], row["api"]
        qidx = qid2idx[qid]
        ordered_apiids = top_k_ids[qidx]
        assert apiid in ordered_apiids
        # find the rank of the docid
        rank = np.where(ordered_apiids == apiid)[0][0] + 1
        ranks.append(rank)
    qd_mapping["rank"] = ranks
    # print("Ranking added to each query-api pair")
    # print(qd_mapping_filtered.head())

    # compute metrics (MRR, Precision@K, Recall@K)
    qd_mapping = qd_mapping.rename(columns={"query": "qid"})
    results = {}
    results["MRR"] = mean_reciprocal_rank(qd_mapping)
    for k in k_list:
        results[f"Precision@{k}"], results[f"Recall@{k}"] = precision_recall_at_k(qd_mapping, k)
    return qd_mapping, results
