import argparse
import json
import os
from dotenv import load_dotenv
from tqdm import tqdm

from agent_system.src.tool_datasets import ToolbenchDataset, APIGenDataset, MetaToolDataset
from agent_system.src.autogen_wrapper import AutogenWrapper
from agent_system.src.tool_retriever import ToolRetriever
from agent_system.src.utils import convert_api_for_registration

def run_agent_system(queries, tools, ds, tool_top_k, autogen_max_chat_round, save_path):
    print(f"Running agent system for {ds.name} dataset")
    print(f"# of queries: {len(queries)}, # of tools: {len(tools)}")
    tool_retriever = ToolRetriever(ds.name)

    # load result file
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            qid2result = json.load(f)
        qid2result = {int(k): v for k, v in qid2result.items()}
    else:
        qid2result = {}
    
    for i, qid in enumerate(tqdm(queries)):
        if qid in qid2result:
            print(f"Query {qid} already exists in result file")
            continue
        query = ds.get_query_by_id(qid)
        gt_apis = ds.get_apis_by_query_id(qid)
        gt_api_ids = ds.get_api_ids_by_query_id(qid)
        answers = ds.get_answers_by_query_id(qid)

        retrieved_api_ids = tool_retriever.call(query_id=qid, k=tool_top_k, candidate_ids=tools)
        retrieved_apis = [ds.get_api_by_id(api_id) for api_id in retrieved_api_ids]
        retrieved_apis = [convert_api_for_registration(ds.name, api) for api in retrieved_apis]

        print(f"Query {qid}: {query}")
        print(f"Number of ground truth APIs: {len(gt_apis)}")
        print(f"Number of retrieved APIs: {len(retrieved_apis)}")
        print(f"Number of API calls: {len(answers)}")

        # check if correct APIs are retrieved
        correct_api_flag = set(gt_api_ids).issubset(set(retrieved_api_ids))
        print(f"Are correct APIs retrieved?: {correct_api_flag}")
        # print(answers)
        # breakpoint()

        # ==== instantiate autogen wrapper
        autogen_wrapper = AutogenWrapper(max_chat_round=autogen_max_chat_round)
        autogen_wrapper.create(["user", "orchestrator", "tool_executor", "tool_execution_manager"])
        print("AutogenWrapper created successfully")

        # register functions
        autogen_wrapper.register_tools(retrieved_apis)
        
        # TODO handle errors
        try:
            tool_calls, traces, final_response = autogen_wrapper.initiate_chat(user_query=query)
        except Exception as e:
            print(f"Error while running autogen: {e}")
            continue

        # write tool_calls and response to file
        example_result_dict = {
            "retrieved_api_ids": retrieved_api_ids,
            "tool_calls": tool_calls,
            "traces": traces,
            "final_response": final_response,
        }
        qid2result[qid] = example_result_dict

        with open(save_path, "w+") as f:
            json.dump(qid2result, f)
        print(f"Query {qid} written to result")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", type=int, default=80)
    parser.add_argument("--dataset", type=str, default="toolbench", choices=["toolbench", "apigen", "metatool"])
    parser.add_argument("--tool_top_k", type=int, default=20)
    parser.add_argument("--autogen_max_chat_round", type=int, default=50)
    parser.add_argument("--result_dir", type=str, default="results")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    load_dotenv(".env")
    args = parse_args()

    if args.dataset == "toolbench":
        ds = ToolbenchDataset()
    elif args.dataset == "apigen":
        ds = APIGenDataset()
    elif args.dataset == "metatool":
        ds = MetaToolDataset()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    # # test tools
    # autogen_wrapper = AutogenWrapper(max_chat_round=args.autogen_max_chat_round)
    # autogen_wrapper.create(["user", "orchestrator", "tool_executor", "tool_execution_manager"])
    # print("AutogenWrapper created successfully")
    # tools = ds.get_api_data_with_query()
    # for tool_id, tool in tqdm(tools.items()):
    #     converted_tool = convert_api_for_registration(args.dataset, tool)
    #     autogen_wrapper.register_tools([converted_tool])
    # exit()

    
    # retrieve tools
    # TODO consider merging this into autogen
    tool_retriever = ToolRetriever(args.dataset)

    # result path
    os.makedirs(args.result_dir, exist_ok=True)
    result_file_path = os.path.join(args.result_dir, f"{args.dataset}_api{args.tool_top_k}.json")
    print(f"Writing result to {result_file_path}")

    # load result file
    if os.path.exists(result_file_path):
        with open(result_file_path, "r") as f:
            result_dict = json.load(f)
        result_dict = {int(k): v for k, v in result_dict.items()}
    else:
        result_dict = {}

    for qid in tqdm(ds.get_id2query()):
        query = ds.get_query_by_id(qid)
        gt_apis = ds.get_apis_by_query_id(qid)
        gt_api_ids = ds.get_api_ids_by_query_id(qid)
        answers = ds.get_answers_by_query_id(qid)

        # let's process multi-step queries first
        if len(gt_api_ids) <= 1:
            continue

        if qid in result_dict:
            print(f"Query {qid} already exists in result file")
            continue
        retrieved_api_ids = tool_retriever.call(query_id=qid, k=args.tool_top_k)
        retrieved_apis = [ds.get_api_by_id(api_id) for api_id in retrieved_api_ids]
        retrieved_apis = [convert_api_for_registration(args.dataset, api) for api in retrieved_apis]

        print(f"Query {qid}: {query}")
        print(f"Number of ground truth APIs: {len(gt_apis)}")
        print(f"Number of retrieved APIs: {len(retrieved_apis)}")
        print(f"Number of API calls: {len(answers)}")

        # check if correct APIs are retrieved
        correct_api_flag = set(gt_api_ids).issubset(set(retrieved_api_ids))
        print(f"Are correct APIs retrieved?: {correct_api_flag}")
        # print(answers)
        # breakpoint()

        # ==== instantiate autogen wrapper
        autogen_wrapper = AutogenWrapper(max_chat_round=args.autogen_max_chat_round)
        autogen_wrapper.create(["user", "orchestrator", "tool_executor", "tool_execution_manager"])
        print("AutogenWrapper created successfully")

        # register functions
        autogen_wrapper.register_tools(retrieved_apis)
        
        # TODO handle errors
        try:
            tool_calls, traces, final_response = autogen_wrapper.initiate_chat(user_query=query)
        except Exception as e:
            print(f"Error while running autogen: {e}")
            breakpoint()
            continue

        # write tool_calls and response to file
        example_result_dict = {
            "retrieved_api_ids": retrieved_api_ids,
            "tool_calls": tool_calls,
            "traces": traces,
            "final_response": final_response,
        }
    
        # write to result file
        result_dict[qid] = example_result_dict
        with open(result_file_path, "w+") as f:
            json.dump(result_dict, f)
        print(f"Query {qid} written to result")
