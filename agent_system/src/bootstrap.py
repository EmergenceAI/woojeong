import argparse
import pickle
import os
import random
import numpy as np
from dotenv import load_dotenv
from itertools import chain
from tqdm import tqdm
from agent_system.src.tool_datasets import ToolbenchDataset, APIGenDataset, MetaToolDataset
from agent_system.src.main import run_agent_system


def sample_lists(lists, target_sum=10, repeats=20, seed=42):
    random.seed(seed)
    sampled_elements = []
    
    for _ in range(repeats):
        remaining_lists = lists.copy()
        sampled_list = []
        current_sum = 0
        
        while current_sum < target_sum and remaining_lists:
            list_choice = random.choice(remaining_lists)
            list_length = len(list_choice)
            
            if current_sum + list_length == target_sum - 1:
                continue
            if current_sum + list_length <= target_sum:
                sampled_list.append(list_choice)
                current_sum += list_length
                remaining_lists.remove(list_choice)
        
        samples = set(chain(*sampled_list))
        sampled_elements.append(samples)
    
    return sampled_elements


def generate_split_info(ds, n_api_list, n_repeat_list):
    grouped_qa_mapping = ds.get_query2apis()
    unique_apis = sorted(ds.get_api_ids_with_query())
    print(f"Unique APIs: {len(unique_apis)}")

    # some apis appear in groups, group them
    api_groups = [set(v) for k, v in grouped_qa_mapping.items()]
    # select unique lists from list of lists
    api_groups = list(set(map(tuple, api_groups)))
    api_groups = list(map(list, api_groups))
    # sort each api group
    api_groups = [sorted(api_group) for api_group in api_groups]
    # sort all api groups by 1. first element, 2. length of list
    api_groups = sorted(api_groups, key=lambda x: (x[0], len(x)))
    print(f"Unique API groups: {len(api_groups)}")

    # split info
    split_info_list = []

    for n_apis, n_repeat in zip(n_api_list, n_repeat_list):
        sampled_apis = sample_lists(api_groups, target_sum=n_apis, repeats=n_repeat, seed=43)
        sampled_queries = []
        for i, result in enumerate(sampled_apis):
            result = result
            # print(f"Sample {i+1}: {result}")
            current_queries = [q for q, apis in grouped_qa_mapping.items() if all([api in result for api in apis])]
            sampled_queries.append(current_queries)
            # print(len(current_queries))
        split_info_list.append({
            "n_apis": n_apis,
            "n_repeat": n_repeat,
            "apis": sampled_apis,
            "queries": sampled_queries
        })
        print(f"# of apis: {n_apis}, # of trials: {n_repeat}, mean # of queries {np.mean([len(q) for q in sampled_queries])}")
    return split_info_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toolbench", choices=["toolbench", "apigen", "metatool"])
    parser.add_argument("--multi_step", action="store_true")
    parser.add_argument("--tool_top_k", type=int, default=20)
    parser.add_argument("--autogen_max_chat_round", type=int, default=50)
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    load_dotenv(".env")
    args = parse_args()

    if args.dataset == "toolbench":
        ds = ToolbenchDataset(multi_step=args.multi_step)
    elif args.dataset == "apigen":
        ds = APIGenDataset(multi_step=args.multi_step)
    elif args.dataset == "metatool":
        ds = MetaToolDataset()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    # # of apis and how many times to repeat
    n_api_list = [10, 50, 100, 200, 500, 1000, 2000] # 2000, 5000, len(unique_apis)]
    n_repeat_list = [3] * len(n_api_list)  # [20, 20, 20, 20, 10, 10, 2, 1]

    # generate split info for bootstrapping
    split_info_list = generate_split_info(ds, n_api_list, n_repeat_list)
    # # or load
    # split_info_list = pickle.load(open(f'/Users/woojeong/Desktop/woojeong/toolbench_analysis/data/split_info_list_{args.dataset}.pkl', 'rb'))
    print(f"Loaded split info list with {len(split_info_list)} splits")

    # NOTE very slow, consider running in parallel
    for split_info in split_info_list:
        print(f"==== # of apis: {split_info['n_apis']}")
        n_apis = split_info["n_apis"]
        n_repeat = split_info["n_repeat"]
        
        for i in range(n_repeat):
            file_path = f'{args.result_dir}/{args.dataset}_api{n_apis}_top{args.tool_top_k}_trial{i}_sim_{os.getenv("SIMULATOR_MODEL_NAME")}.json'
            apis = split_info["apis"][i]
            queries = split_info["queries"][i]

            print(f"Trial {i+1}/{n_repeat}, # of queries: {len(queries)}")
            print(f"Result file path: {file_path}")
            if args.tool_top_k > len(apis):
                raise ValueError(f"tool_top_k ({args.tool_top_k}) should be less than or equal the size of api pool ({len(apis)})")
            run_agent_system(queries, apis, ds, args.tool_top_k, args.autogen_max_chat_round, file_path)
