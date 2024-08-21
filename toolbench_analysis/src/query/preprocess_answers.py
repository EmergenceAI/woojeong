import os
import json
import argparse
import pickle
from natsort import natsorted
from dotenv import load_dotenv
from tqdm import tqdm
from agent_system.src.tool_datasets import ToolbenchDataset
from agent_system.src.utils import convert_api_name


def extrace_tool_calls_from_traces(traces):
    tool_calls = []
    for i in range(len(traces) - 1):
        trace = traces[i]
        if trace["role"] == "assistant" and "function_call" in trace:
            func_name = trace["function_call"]["name"]
            func_args = trace["function_call"]["arguments"]

            # check if next trace is function call
            next_trace = traces[i + 1]
            if next_trace["role"] == "function":
                assert next_trace["name"] == func_name
                func_output = next_trace["content"]

            tool_calls.append(
                {"name": func_name, "args": func_args, "output": func_output}
            )
    return tool_calls


def match_apis(api_list, api_data_df):
    if len(api_list) == 0:
        return []

    matched_apis = []
    for api in api_list:
        category_name = api["category_name"]
        tool_name = api["tool_name"]
        api_name = api["api_name"]

        api_match = api_data_df[
            (api_data_df["category_name"] == category_name)
            & (api_data_df["tool_name"] == tool_name)
            & (api_data_df["api_name"] == api_name)
        ]
        if len(api_match) != 1:
            return []
        matched_apis.append(api_match.index[0])
    return matched_apis


def parse_final_answer(action_input):
    try:
        json_data = json.loads(action_input, strict=False)
    except:
        json_data = {}
        if '"return_type": "' in action_input:
            if '"return_type": "give_answer"' in action_input:
                return_type = "give_answer"
            elif '"return_type": "give_up_and_restart"' in action_input:
                return_type = "give_up_and_restart"
            else:
                return_type = action_input[
                    action_input.find('"return_type": "')
                    + len('"return_type": "') : action_input.find('",')
                ]
            json_data["return_type"] = return_type
        if '"final_answer": "' in action_input:
            final_answer = action_input[
                action_input.find('"final_answer": "') + len('"final_answer": "') :
            ]
            json_data["final_answer"] = final_answer

    if "return_type" not in json_data.keys():
        json_data["return_type"] = "error"
    if json_data["return_type"] == "give_up_and_restart":
        if "final_answer" not in json_data.keys():
            json_data["final_answer"] = ""
    elif json_data["return_type"] == "give_answer":
        if "final_answer" not in json_data.keys():
            json_data["final_answer"] = ""

    return json_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="G1")
    parser.add_argument("--result_dir", type=str, default="data")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    load_dotenv(".env")
    args = parse_args()

    # load toolbench train data
    toolbench_folder = os.getenv("TOOLBENCH_FOLDER")
    inst_path = os.path.join(
        toolbench_folder, f"data/instruction/{args.subset}_query.json"
    )
    with open(inst_path, "r") as f:
        inst_list = json.load(f)
    inst_queries = [x["query_id"] for x in inst_list]
    inst_dict = {x["query_id"]: x for x in inst_list}
    print(f"Loaded {len(inst_list)} instruction queries")

    # load toolbench answers
    ans_dir = os.path.join(toolbench_folder, f"data/answer/{args.subset}_answer")
    # load all files from ans_dir
    ans_files = natsorted(os.listdir(ans_dir))
    ans_queries = [int(x.split("_")[0]) for x in ans_files]
    print(f"Loaded {len(ans_files)} answer files")
    assert set(ans_queries).issubset(
        set(inst_queries)
    ), "Answer queries not subset of instruction queries"

    # load toolbench ds
    ds = ToolbenchDataset()
    api_data_df = ds.get_api_data_df()

    # parse and filter out invalid answers
    gt_dict = {}

    for qid in tqdm(ans_queries):
        # load inst
        inst_q = inst_dict[qid]
        api_list = inst_q["api_list"]
        relevant_apis = inst_q["relevant APIs"]
        # map api_list to index
        api_ids = match_apis(api_list, api_data_df)
        if len(api_ids) == 0:
            continue

        # load answer file
        with open(os.path.join(ans_dir, f"{qid}_ChatGPT_DFS_woFilter_w2.json")) as f:
            ans = json.load(f)

        # parse info
        query = ans["answer_generation"]["query"]
        if type(query) == list:
            continue
        functions = ans["answer_generation"]["function"]
        functions = [
            f for f in functions if f["name"] != "Finish"
        ]  # remove Finish helper function
        finish_type = ans["answer_generation"]["finish_type"]
        final_answer = parse_final_answer(ans["answer_generation"]["final_answer"]).get(
            "final_answer", ""
        )
        # select valid answers
        if (finish_type == "give_up") or (final_answer == ""):
            continue
        traces = ans["answer_generation"]["train_messages"][-1]
        tool_calls = extrace_tool_calls_from_traces(traces)
        tool_calls = [
            call for call in tool_calls if call["name"] != "Finish"
        ]  # remove Finish helper function
        
        # check if args are valid json
        skip_flag = False
        for tool_call in tool_calls:
            try:
                json.loads(tool_call['args'])
            except:
                print(f"Invalid args: {tool_call['args']}")
                skip_flag = True
                break
        if skip_flag:
            continue

        # match api_ids with functions
        
        name2apiid = {}
        for api_id, func in zip(api_ids, functions):
            api = ds.get_api_by_id(api_id)
            func_name = convert_api_name(api["tool_name"], api["api_name"])
            if func_name == func["name"]:
                func["api_id"] = api_id
                name2apiid[func_name] = api_id
            else:
                # function name mismatch, skip this query
                print(f"Function name mismatch: {func_name} != {func['name']}")
                skip_flag = True
                break

        if skip_flag:
            continue

        # match tool_calls with api_ids
        for tool_call in tool_calls:
            if tool_call["name"] in name2apiid:
                tool_call["api_id"] = name2apiid[tool_call["name"]]
            else:
                # tool call not matching, continue
                print(f"Tool call {tool_call['name']} not found in {name2apiid.keys()}")
                skip_flag = True
                break
        if skip_flag:
            continue

        # sanity check
        assert len(api_list) == len(functions)

        gt_dict[qid] = {
            "query": query,
            "final_answer": final_answer,
            "traces": traces,
            "functions": functions,
            "tool_calls": tool_calls,
            "api_ids": api_ids,
        }

    # save results
    print(f"Found {len(gt_dict)} valid answers")
    save_path = os.path.join(args.result_dir, f"{args.subset}_gt.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(gt_dict, f)
    print(f"Saved results to {save_path}")
    
