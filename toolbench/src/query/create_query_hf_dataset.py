import json
import logging
import os
import argparse
import pandas as pd
from natsort import natsorted
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, default=".env")
    parser.add_argument(
        "--out_local_csv_path", type=str, default="data/subset_query_data.csv"
    )
    parser.add_argument(
        "--hf_dataset_name", type=str, default="MerlynMind/toolbench_query"
    )
    parser.add_argument("--subset", type=str, default="g1")
    parser.add_argument("--push_to_hub", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_dotenv(args.env_path)

    # load instruction
    toolbench_dir = os.getenv("TOOLBENCH_FOLDER")
    inst_path = os.path.join(
        toolbench_dir, f"data/instruction/{args.subset.upper()}_query.json"
    )

    with open(inst_path) as f:
        inst = json.load(f)
    logging.info(f"Loaded instruction from {inst_path}")
    logging.info(f"Total number of instruction: {len(inst)}")
    inst_df = pd.json_normalize(inst)
    logging.info(f"Instruction dataframe shape: {inst_df.shape}")

    # load answer
    ans_dir = os.path.join(toolbench_dir, f"data/answer/{args.subset.upper()}_answer")
    ans_files = os.listdir(ans_dir)
    ans_files = natsorted(ans_files, key=lambda y: y.lower())
    logging.info(f"Loaded {len(ans_files)} answer files from {ans_dir}")

    # add query_id to each answer
    logging.info("Adding query_id to each answer dictionary")
    answers = []
    for ans_filename in tqdm(ans_files):
        ans_path = os.path.join(ans_dir, ans_filename)
        with open(ans_path) as f:
            ans = json.load(f)
        qid = int(ans_filename.split("_")[0])
        ans["query_id"] = qid
        answers.append(ans)

    # extract necessary information from answers
    logging.info("Extracting necessary information from answers")
    info_list = []
    for ans in tqdm(answers):
        query_id = ans["query_id"]
        win = ans["win"]
        valid_data = ans["answer_generation"]["valid_data"]
        final_answer = ans["answer_generation"]["final_answer"]
        finish_type = ans["answer_generation"]["finish_type"]
        function = ans["answer_generation"]["function"]
        train_messages = ans["answer_generation"]["train_messages"]
        query = ans["answer_generation"]["query"]

        # make above as a row
        row = {
            "query_id": query_id,
            "win": win,
            "valid_data": valid_data,
            "final_answer": final_answer,
            "finish_type": finish_type,
            "function": function,
            "train_messages": train_messages,
            "query_y": query,
        }
        info_list.append(row)
    answer_df = pd.DataFrame(info_list)
    logging.info(f"Answer dataframe shape: {answer_df.shape}")

    # join inst_df and answer_df by query_id
    # inst_df has more rows, so we need to use left join
    # extract where answer field is not null
    inst_answer_df = inst_df.merge(answer_df, on="query_id", how="left")
    inst_answer_df = inst_answer_df[inst_answer_df["query_y"].notnull()]
    logging.info(f"Joined dataframe shape: {inst_answer_df.shape}")
    logging.info(
        f"{inst_df.shape[0] - inst_answer_df.shape[0]} rows are in inst_df but not in answer_df"
    )

    # check whether query and query_y are the same, if not, raise error
    assert (
        inst_answer_df["query"] == inst_answer_df["query_y"]
    ).all(), "query and query_y are not the same"
    inst_answer_df.drop(columns=["query_y"], inplace=True)

    # check whether all data are valid
    assert (inst_answer_df["valid_data"] == True).all(), "Some data are invalid"
    inst_answer_df.drop(columns=["valid_data"], inplace=True)

    # some preprocessing
    inst_answer_df.rename(columns={"relevant APIs": "relevant_apis"}, inplace=True)
    inst_answer_df = inst_answer_df[
        [
            "query_id",
            "query",
            "relevant_apis",
            "api_list",
            "function",
            "win",
            "final_answer",
            "finish_type",
            "train_messages",
        ]
    ]

    # save to local csv
    args.out_local_csv_path = args.out_local_csv_path.replace("subset", args.subset)
    logging.info(f"Saving to {args.out_local_csv_path}")
    inst_answer_df.to_csv(args.out_local_csv_path, index=False)
    logging.info(inst_answer_df.head())

    if not args.push_to_hub:
        logging.info("Skipping push to Hugging Face dataset")
        exit()

    # these columns should be mapped to "string" to avoid errors when converting to hf dataset
    str_cols = [
        "relevant_apis",
        "api_list",
        "function",
        "final_answer",
        "train_messages",
        "finish_type",
    ]
    for col in inst_answer_df.columns:
        if col in inst_answer_df.columns:
            inst_answer_df[col] = inst_answer_df[col].astype(str)
    dataset = Dataset.from_pandas(inst_answer_df)
    dataset.push_to_hub(args.hf_dataset_name, args.subset, token=os.getenv("HF_TOKEN"))
    logging.info(
        f"Pushed to Hugging Face dataset: {args.hf_dataset_name}/{args.subset}"
    )
