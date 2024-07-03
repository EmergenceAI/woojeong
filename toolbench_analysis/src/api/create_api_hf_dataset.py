import json
import logging
import os
import numpy as np
import argparse
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)


def json_to_df(apis):
    categories_df = []
    for category in apis:
        logging.info(f"category: {category}")
        tools = apis[category]
        if len(tools) == 0:
            print(category, "is empty")
            continue

        # convert tools to dataframe
        tools_df = []
        for tool_name in tools:
            # print(f"tool: {tool_name}")
            tool = tools[tool_name]
            df = pd.json_normalize(tool, meta=tool.keys())
            tools_df.append(df)

        # concat all tools in the category
        tools_df = pd.concat(tools_df)
        tools_df["category_name"] = category
        categories_df.append(tools_df)
    apis_df = pd.concat(categories_df).reset_index(drop=True)
    return apis_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, default=".env")
    parser.add_argument("--api_json_path", type=str, default="data/api_data.json")
    parser.add_argument("--out_local_csv_path", type=str, default="data/api_data.csv")
    parser.add_argument(
        "--hf_dataset_name", type=str, default="MerlynMind/toolbench_api"
    )
    parser.add_argument("--push_to_hub", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_dotenv(args.env_path)

    # load and convert json -> df
    with open(args.api_json_path, "r") as f:
        apis = json.load(f)
    logging.info(f"Loaded {len(apis)} categories")
    logging.info(f"Creating dataframe from json")
    apis_df = json_to_df(apis)
    logging.info(f"# of tools: {apis_df.shape[0]}")

    # delete useless columns
    del_cols = ["score.__typename", "score", "product_id", "name",]
    for col in del_cols:
        if col in apis_df.columns:
            del apis_df[col]

    # add tool_ prefix
    apis_df.columns = [
        "tool_name",
        "tool_description",
        "tool_title",
        "tool_pricing",
        "tool_home_url",
        "tool_host",
        "api_list",
        "tool_avg_service_level",
        "tool_avg_latency",
        "tool_avg_success_rate",
        "tool_popularity_score",
        "tool_name_standardized",
        "category_name",
    ]

    # flatten api_list column
    apis_df = apis_df.explode("api_list").reset_index(drop=True)
    # from api_list column, make elements in dict as separate columns
    api_info_only = apis_df["api_list"].apply(pd.Series)
    # add "api_" prefix to all columns
    api_info_only.columns = [f"api_{col}" for col in api_info_only.columns]
    apis_df = pd.concat(
        [apis_df.drop(["api_list"], axis=1), api_info_only],
        axis=1,
    )
    # bring "category_name" column to the front
    apis_df = apis_df[
        ["category_name"] + [col for col in apis_df.columns if col != "category_name"]
    ]

    # handle missing values
    fillna_cols = ["tool_name_standardized", "api_body", "api_headers", "api_schema", "api_convert_code", "api_test_endpoint"]
    for col in fillna_cols:
        apis_df[col] = apis_df[col].replace("", None)
        apis_df[col] = apis_df[col].replace(np.nan, None)
        apis_df[col] = apis_df[col].replace("nan", None)

    # save to local csv
    logging.info(f"Final dataframe shape: {apis_df.shape}")
    logging.info(f"Saving to {args.out_local_csv_path}")
    apis_df.to_csv(args.out_local_csv_path, index=False)
    logging.info(f"Saved to {args.out_local_csv_path}")

    # convert to hf dataset
    if args.push_to_hub:
        # these columns should be mapped to "string" to avoid errors when converting to hf dataset
        str_cols = [
            "api_required_parameters",
            "api_optional_parameters",
            "api_body",
            "api_headers",
            "api_schema",
            "api_test_endpoint",
        ]
        for col in str_cols:
            if col in apis_df.columns:
                apis_df[col] = apis_df[col].astype(str)

        dataset = Dataset.from_pandas(apis_df)
        dataset.push_to_hub(args.hf_dataset_name, token=os.getenv("HF_TOKEN"))
        logging.info(f"Pushed to Hugging Face dataset: {args.hf_dataset_name}")
    else:
        logging.info("Skipping push to Hugging Face dataset")
    print(dataset)
