import os
import pandas as pd
import ast
import numpy as np
from datasets import load_dataset


def load_query_api_mapping(local_file_path="data/query_api_mapping.csv"):
    """
    Load query-api mapping data from local file or hf dataset.

    Args:
        local_file_path (str): path to local file

    Returns:
        df (pd.DataFrame): query-api mapping data
        id2doc (dict): mapping from query id to api id
        id2query (dict): mapping from api id to query id
    """
    if os.path.exists(local_file_path):
        print(f"Loading query-api mapping from local file: {local_file_path}")
        df = pd.read_csv(local_file_path)
    else:
        print(f"Loading from hf dataset: MerlynMind/toolbench_query_api_mapping")
        # load from hf dataset
        dataset_dict = load_dataset(
            "MerlynMind/toolbench_query_api_mapping", "g1", token=os.getenv("HF_TOKEN")
        )
        train_df = dataset_dict["train"].to_pandas()
        test_df = dataset_dict["test"].to_pandas()
        # add split column
        train_df["split"] = "train"
        test_df["split"] = "test"
        df = pd.concat([train_df, test_df])

    # make id2doc and id2query dicts
    id2doc = {}
    id2query = {}
    for i, row in df.iterrows():
        query_id = row["qid"]
        api_id = row["docid"]
        if query_id not in id2query:
            id2query[query_id] = row["query"]
        if api_id not in id2doc:
            id2doc[api_id] = row["doc"]
    return df, id2doc, id2query


def load_api_data(local_file_path="data/api_data.csv"):
    """
    Load api data from local file or hf dataset.

    Args:
        local_file_path (str): path to local file

    Returns:
        df (pd.DataFrame): api data
    """
    if os.path.exists(local_file_path):
        print(f"Loading api data from local file: {local_file_path}")
        df = pd.read_csv(local_file_path)
    else:
        print(f"Loading from hf dataset: MerlynMind/toolbench_api")
        ds = load_dataset("MerlynMind/toolbench_api", token=os.getenv("HF_TOKEN"))[
            "train"
        ]
        df = ds.to_pandas()

    # post-processing
    # some columns are mapped to "string" to avoid errors when converting to hf dataset
    # convert them back to their original types
    str_cols = [
        "api_required_parameters",
        "api_optional_parameters",
        # "api_body",
        # "api_headers",
        # "api_schema",
        # "api_test_endpoint",
    ]

    def eval(x):
        try:
            return ast.literal_eval(x)
        except:
            return x

    for col in str_cols:
        if col in df.columns:
            print(f"processing '{col}' column")
            df[col] = df[col].apply(eval)
    return df


def load_query_data(split="g1", local_file_path="data/g1_query_data.csv"):
    """
    Load query data from local file or hf dataset.

    Args:
        split (str): dataset split, one of ['g1', 'g2', 'g3']
        local_file_path (str): path to local file

    Returns:
        df (pd.DataFrame): query data
    """
    if split not in ["g1", "g2", "g3"]:
        raise ValueError("split must be one of ['g1', 'g2', 'g3']")
    local_file_path = local_file_path.replace("g1", split)
    if os.path.exists(local_file_path):
        print(f"Loading query data from local file: {local_file_path}")
        df = pd.read_csv(local_file_path)
    else:
        print(f"Loading from hf dataset: MerlynMind/toolbench_query")
        ds = load_dataset(
            "MerlynMind/toolbench_query", split, token=os.getenv("HF_TOKEN")
        )["train"]
        df = ds.to_pandas()

    # post-processing
    # some columns are mapped to "string" to avoid errors when converting to hf dataset
    # convert them back to their original types
    # this may take a few minutes
    df["relevant_apis"] = df["relevant_apis"].apply(ast.literal_eval)
    df["api_list"] = df["api_list"].apply(ast.literal_eval)
    df["function"] = df["function"].apply(ast.literal_eval)
    df["train_messages"] = df["train_messages"].apply(ast.literal_eval)
    return df
