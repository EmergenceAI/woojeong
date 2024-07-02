import json
import logging
import os
import argparse
import pickle
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import Dataset, DatasetDict

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)


def map_id_to_doc(documents_df, toolbench_string=False):
    """
    Process API documents as in toolbench paper
    """
    id2doc = {}
    for row in documents_df.itertuples():
        if toolbench_string:
            doc = json.loads(row.document_content)
            doc = (
                (doc.get("category_name", "") or "")
                + ", "
                + (doc.get("tool_name", "") or "")
                + ", "
                + (doc.get("api_name", "") or "")
                + ", "
                + (doc.get("api_description", "") or "")
                + ", required_params: "
                + json.dumps(doc.get("required_parameters", ""))
                + ", optional_params: "
                + json.dumps(doc.get("optional_parameters", ""))
                + ", return_schema: "
                + json.dumps(doc.get("template_response", ""))
            )
        else:
            doc = row.document_content
        id2doc[row.docid] = doc
    return id2doc


def get_query_doc_mappings(data_dir, split="train"):
    """
    load query and query-doc mappings
    """
    queries_df = pd.read_csv(
        os.path.join(data_dir, f"{split}.query.txt"), sep="\t", names=["qid", "query"]
    )
    query_doc_mapping_df = pd.read_csv(
        os.path.join(data_dir, f"qrels.{split}.tsv"),
        sep="\t",
        names=["qid", "useless", "docid", "label"],
    )

    # id2query dict
    # group by qid and list ["query"] column in list
    unique_queries = queries_df.groupby("qid").agg({"query": set}).reset_index()
    # assert all "query" column length is 1
    assert all(unique_queries["query"].apply(len) == 1)
    # get the first element of the list
    unique_queries["query"] = unique_queries["query"].apply(lambda x: x.pop())
    # convert to {qid: query} dictionary
    id2query = unique_queries.set_index("qid")["query"].to_dict()

    # process query_doc_mapping_df
    # sort labels_df by qid, docid
    query_doc_mapping_df = query_doc_mapping_df.sort_values(
        by=["qid", "docid"]
    ).reset_index(drop=True)

    # remove useless, label columns
    query_doc_mapping_df = query_doc_mapping_df.drop(columns=["useless", "label"])

    # get unique doc_ids
    doc_ids = query_doc_mapping_df["docid"].unique()

    return id2query, doc_ids, query_doc_mapping_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, default=".env")
    parser.add_argument(
        "--out_local_csv_path", type=str, default="data/query_api_mapping.csv"
    )
    parser.add_argument(
        "--hf_dataset_name", type=str, default="MerlynMind/toolbench_query_api_mapping"
    )
    parser.add_argument("--push_to_hub", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_dotenv(args.env_path)

    # load api documents
    toolbench_dir = os.getenv("TOOLBENCH_FOLDER")
    data_dir = os.path.join(
        toolbench_dir, f"data/retrieval/G1/"
    )  # only G1 is available
    api_doc_df = pd.read_csv(os.path.join(data_dir, "corpus.tsv"), sep="\t")
    logging.info(f"Loaded API documents from {data_dir}")
    logging.info(f"Total # of API calls: {len(api_doc_df)}")

    # extract unique apis
    id2doc = map_id_to_doc(api_doc_df)
    logging.info(f"Number of unique APIs: {len(id2doc)}")

    # load query-doc mappings
    train_id2query, train_doc_ids, train_qd_mapping = get_query_doc_mappings(
        data_dir, split="train"
    )
    test_id2query, test_doc_ids, test_qd_mapping = get_query_doc_mappings(
        data_dir, split="test"
    )
    logging.info("# of queries, # of api calls")
    logging.info(f"train: {len(train_id2query)}, {len(train_qd_mapping)}")
    logging.info(f"test: {len(test_id2query)}, {len(test_qd_mapping)}")

    # overlapping qids from test
    overlap_qid = set(pd.unique(train_qd_mapping["qid"])) & set(
        pd.unique(test_qd_mapping["qid"])
    )
    test_id2query = {
        qid: query for qid, query in test_id2query.items() if qid not in overlap_qid
    }
    test_qd_mapping = test_qd_mapping[~test_qd_mapping["qid"].isin(overlap_qid)]
    logging.info("# of queries, # of api calls after removing overlapping qids")
    logging.info(f"test: {len(test_id2query)}, {len(test_qd_mapping)}")

    # check train_doc_ids and test_doc_ids are in id2doc keys
    assert set(train_doc_ids) <= set(id2doc.keys())
    assert set(test_doc_ids) <= set(id2doc.keys())

    # merge train and test
    id2query = {**train_id2query, **test_id2query}
    # add split column
    train_qd_mapping["split"] = "train"
    test_qd_mapping["split"] = "test"
    qd_mapping = pd.concat([train_qd_mapping, test_qd_mapping])
    logging.info("Merged train and test query-doc mappings")
    logging.info(f"Total # of queries: {len(id2query)}")
    logging.info(f"Total # of APIs: {len(id2doc)}")
    logging.info(f"Total # of query-doc mappings: {len(qd_mapping)}")

    # id2query dict to df
    id2query_df = pd.DataFrame(id2query.items(), columns=["qid", "query"])
    id2doc = pd.DataFrame(id2doc.items(), columns=["docid", "doc"])

    # join qd_mapping with id2query and id2doc
    qd_mapping = qd_mapping.merge(id2query_df, on="qid", how="left")
    qd_mapping = qd_mapping.merge(id2doc, on="docid", how="left")

    # save to local csv
    # make dir if not exists
    # os.makedirs(args.out_local_csv_dir, exist_ok=True)
    # with open(os.path.join(args.out_local_csv_dir, "id2query.pkl"), "wb") as f:
    #     pickle.dump(id2query, f)
    # with open(os.path.join(args.out_local_csv_dir, "id2doc.pkl"), "wb") as f:
    #     pickle.dump(id2doc, f)
    qd_mapping.to_csv(args.out_local_csv_path, index=False)
    logging.info(f"Saved qd_mapping to {args.out_local_csv_path}")

    # push to huggingface hub
    if args.push_to_hub:
        # split train/test
        train_qd_mapping = (
            qd_mapping[qd_mapping["split"] == "train"]
            .reset_index(drop=True)
            .drop(columns=["split"])
        )
        test_qd_mapping = (
            qd_mapping[qd_mapping["split"] == "test"]
            .reset_index(drop=True)
            .drop(columns=["split"])
        )
        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_pandas(train_qd_mapping),
                "test": Dataset.from_pandas(test_qd_mapping),
            }
        )
        dataset_dict.push_to_hub(
            args.hf_dataset_name, "g1", token=os.getenv("HF_TOKEN")
        )
        logging.info(f"Pushed to Hugging Face dataset: {args.hf_dataset_name}/g1")
