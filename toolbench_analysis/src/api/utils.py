import json
import logging
import os
import tiktoken
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import openai
import numpy as np

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)


# Custom thread pool executor for gpt requests
class CustomThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers):
        super().__init__(max_workers=max_workers)

    def submit(self, fn, *args, **kwargs):
        return super().submit(fn, *args, **kwargs)


def store_api_data(api_dict, output_path):
    """Stores the API data in a JSON file.

    Args:
        api_dict (dict): The dictionary containing the API data.
        output_path (str): The path to the output JSON file.
    """
    with open(output_path, "w") as json_file:
        json.dump(api_dict, json_file)


def read_api_data(input_path):
    """Reads the API data from a JSON file.

    Args:
        input_path (str): The path to the input JSON file.

    Returns:
        dict: A dictionary containing the API data.
    """
    try:
        with open(input_path, "r") as json_file:
            return json.load(json_file)
    except Exception as e:
        logging.error(f"Error reading API data: {e}")
        return None


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_gpt_response(messages, model="gpt-4-turbo-preview", temperature=0):
    logging.info("Getting gpt response ...")
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, stream=False
    )
    logging.info(f"GPT response: {response}")
    try:
        output = response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error getting gpt response: {e}")
        output = ""
    return output


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_gpt_response_multithread(messages, model="gpt-4-turbo-preview", temperature=0.0):
    counter_api_request = 0
    counter_tokens = 0
    lock = threading.Lock()

    openai.api_key = os.environ.get('OPENAI_API_KEY')

    timeout_seconds = 60  # Set the timeout to 60 seconds (1 minute)

    start_time = time.time()
    done = False
    user_prompt = messages[-1]["content"] if messages[-1]["role"] == "user" else messages[-2]["content"]

    token_count = num_tokens_from_string(user_prompt, "cl100k_base")

    with lock:
        while not done and (time.time() - start_time) < timeout_seconds:
            print(counter_api_request)
            if counter_api_request >= 100 or counter_tokens + token_count >= 20000:

                time_diff = 60 - (time.time() - start_time)
                if time_diff > 0:
                    print(f'........user_msg is {user_prompt}')
                    print('Waiting to avoid hitting rate and token limits per minute.')
                    time.sleep(time_diff)
                    counter_api_request = 1
                    counter_tokens = token_count
                    start_time = time.time()
                else:
                    counter_api_request += 1
                    counter_tokens += token_count

            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    timeout=120
                )
                # print(f'response is {response}')
            except Exception as e:
                logging.info(f"Exception encountered {e}, going to sleep")
                time.sleep(60)
                continue
            done = True

    if not done:
        logging.info("Timeout exceeded, stopping the loop.")
    gpt_msg = response.choices[0].message.content
    total_tokens_usage = response.usage.total_tokens
    return gpt_msg, total_tokens_usage


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
