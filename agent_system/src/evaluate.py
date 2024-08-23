import argparse
import json
import os
import re
import logging
from dotenv import load_dotenv
from utils import compare_tool_calls
from enum import Enum
from collections import Counter
from itertools import chain

from agent_system.src.tool_datasets import ToolbenchDataset, APIGenDataset, MetaToolDataset
from agent_system.src.prompts import EVALUATION_PROMPT
from agent_system.src.openai_utils import CustomThreadPoolExecutor, get_gpt_response_multithread


class AnswerStatus(Enum):
    Unsure = "Unsure"
    Unsolved = "Unsolved"
    Solved = "Solved"


def parse_eval_output(output):
    # Case 1: Try to handle it as a JSON string
    try:
        # Remove any ```json wrapping
        output = output.strip('```')
        return json.loads(output)
    except json.JSONDecodeError:
        pass

    # Case 2: Use regular expressions to extract content and answer_status
    content = None
    answer_status = None

    # Pattern for the format "**Content**: ...\n**Answer Status**:..."
    pattern_case_1 = r'\*\*Content\*\*: (.*?)\n\*\*Answer Status\*\*: (.*)'
    match = re.search(pattern_case_1, output)
    if match:
        content = match.group(1)
        answer_status = match.group(2)
    else:
        # Pattern for the format "Content: ...\nAnswer Status:..."
        pattern_case_2 = r'Content: (.*?)\nAnswer Status: (.*)'
        match = re.search(pattern_case_2, output)
        if match:
            content = match.group(1)
            answer_status = match.group(2)
        else:
            pass

    # Case 3: find "Solved", "Unsolved", "Unsure" in the output
    if content is None or answer_status is None:
        if "Solved" in output:
            content = "Unknown"
            answer_status = AnswerStatus.Solved
        elif "Unsolved" in output:
            content = "Unknown"
            answer_status = AnswerStatus.Unsolved
        elif "Unsure" in output:
            content = "Unknown"
            answer_status = AnswerStatus.Unsure
    else:
        print(output)
        content = "reponse parsing failed"
        answer_status = AnswerStatus.Unsure

    return {
        "content": content,
        "answer_status": answer_status
    }


def calculate_retrieval_metrics(gt_apis, retrieved_apis, keep_duplicates=False):
    if not keep_duplicates:
        # ignore multiple calls to the same API
        gt_apis = list(set(gt_apis))
        retrieved_apis = list(set(retrieved_apis))
    # for aggregation, return counts
    tp = len(list((Counter(gt_apis) & Counter(retrieved_apis)).elements()))
    fn = len(list((Counter(gt_apis) - Counter(retrieved_apis)).elements()))
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    n_gt = len(gt_apis)

    assert tp + fn == n_gt, f"TP: {tp}, FN: {fn}, n_gt: {n_gt}"

    return n_gt, tp, fn, recall


def calculate_tool_metrics(gt_calls, tool_calls):
    # flatten tool_calls
    tool_calls = list(chain(*tool_calls))
    tool_calls = [call['function'] for call in tool_calls]

    # check whether the correct tools are called
    # ignore the order of calls, same tool called multiple times
    gt_apis = [call["name"] for call in gt_calls]
    tool_apis = [call["name"] for call in tool_calls]

    # each calls are considered as unique
    tool_gt_count, tool_tp, tool_fn, tool_recall = calculate_retrieval_metrics(gt_apis, tool_apis, keep_duplicates=True)

    # for correct tools, check whether the correct parameters are called
    correct_call_count = 0
    for gt_call in gt_calls:
        # load calls for the same tool
        gt_args = gt_call["arguments"]
        pred_args = [json.loads(call["arguments"]) for call in tool_calls if call["name"] == gt_call["name"]]
        if len(pred_args) == 0:
            continue
        if gt_args in pred_args:
            correct_call_count += 1
        else:
            print(f"Ground truth args: {gt_args}")
            print(f"Predicted args: {pred_args}")

    return tool_gt_count, tool_tp, tool_fn, tool_recall, correct_call_count

def check_is_solved(query, traces, final_response):
    # borrowed from toolbench
    # empty situation
    if final_response=='':
        return AnswerStatus.Unsolved, "Empty final answer"
    
    # check whether the final answer solved the query
    prompt = EVALUATION_PROMPT["CHECK_ANSWER_STATUS"]
    prompt = prompt.replace("{query}", query)
    prompt = prompt.replace("{answer}", final_response)
    user_prompt = {"role": "user", "content": prompt}
    response, _ = get_gpt_response_multithread(
        messages=[user_prompt], 
        model=os.getenv("EVAL_MODE_NAME"),
    )

    # parse response
    ret = parse_eval_output(response)
    answer_status = AnswerStatus(ret['answer_status'])
    
    # if unsure, check solution path
    if answer_status == AnswerStatus.Unsure:
        print("Unsure by the final answer... Checking solution path")
        # detailed check here
        prompt = EVALUATION_PROMPT["PARSE_ANSWER_STATUS"]
        prompt = prompt.replace("{query}", query)
        prompt = prompt.replace("{answer}", json.dumps(traces))
        user_prompt = {"role": "user", "content": prompt}
        response, _ = get_gpt_response_multithread(
            messages=[user_prompt], 
            model=os.getenv("EVAL_MODE_NAME"),
        )
        ret = parse_eval_output(response)
        answer_status = AnswerStatus(ret['answer_status'])

    return answer_status, ret['content']


def evaluate(eval_result_dict, args, qid, ds, result):
    if qid in eval_result_dict and not args.overwrite_all:
        eval_result = eval_result_dict[qid]
    else:
        eval_result = {}
    
    print(f"Evaluating query {qid}")
    query = ds.get_query_by_id(qid)
    gt_apis = ds.get_apis_by_query_id(qid)
    gt_api_ids = ds.get_api_ids_by_query_id(qid)
    answers = ds.get_answers_by_query_id(qid)

    retrieved_api_ids = result["retrieved_api_ids"]
    tool_calls = result["tool_calls"]
    traces = result["traces"]
    final_response = result["final_response"]

    # get values from eval_result if already computed
    retrieval_gt_count = eval_result.get("retrieval_gt_count", None)
    retrieval_tp = eval_result.get("retrieval_tp", None)
    retrieval_fn = eval_result.get("retrieval_fn", None)
    retrieval_recall = eval_result.get("retrieval_recall", None)
    tool_gt_count = eval_result.get("tool_gt_count", None)
    tool_tp = eval_result.get("tool_tp", None)
    tool_fn = eval_result.get("tool_fn", None)
    tool_recall = eval_result.get("tool_recall", None)
    tool_correct_call_count = eval_result.get("tool_call_correct_count", None)
    answer_status = eval_result.get("answer_status", None)
    reason = eval_result.get("reason", None)
    
    
    # tool retrieval performance
    if args.eval_retrieval and retrieval_gt_count is None:
        retrieval_gt_count, retrieval_tp, retrieval_fn, retrieval_recall = calculate_retrieval_metrics(gt_api_ids, retrieved_api_ids)
    if args.eval_tool and tool_gt_count is None:
        tool_gt_count, tool_tp, tool_fn, tool_recall, tool_correct_call_count = calculate_tool_metrics(answers, tool_calls)
    if args.eval_solved and answer_status is None:
        answer_status, reason = check_is_solved(query, traces, final_response)
        answer_status = answer_status.value

    eval_result = {
        "retrieval_gt_count": retrieval_gt_count,
        "retrieval_tp": retrieval_tp,
        "retrieval_fn": retrieval_fn,
        "retrieval_recall": retrieval_recall,
        "tool_gt_count": tool_gt_count,
        "tool_tp": tool_tp,
        "tool_fn": tool_fn,
        "tool_recall": tool_recall,
        "tool_call_correct_count": tool_correct_call_count,
        "answer_status": answer_status,
        "reason": reason
    }
    eval_result_dict[qid] = eval_result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toolbench", choices=["toolbench", "apigen", "metatool"])
    parser.add_argument("--tool_top_k", type=int, default=20)
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--eval_retrieval", action="store_true")
    parser.add_argument("--eval_tool", action="store_true")
    parser.add_argument("--eval_solved", action="store_true")
    parser.add_argument("--n_threads", type=int, default=20)
    parser.add_argument("--overwrite_all", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # turn off openai logging
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)

    load_dotenv(".env")
    args = parse_args()

    result_file_path = os.path.join(args.result_dir, f"{args.dataset}_api{args.tool_top_k}.json")
    assert os.path.exists(result_file_path), f"Result file {result_file_path} does not exist"
    print(f"Loading results from {result_file_path}")
    with open(result_file_path, "r") as f:
        result_dict = json.load(f)
    result_dict = {int(k): v for k, v in result_dict.items()}
    print(f"Loaded {len(result_dict)} results")
    
    # count tool calls
    n_tool_calls = [len(result["tool_calls"]) for result in result_dict.values()]
    print(Counter(n_tool_calls))

    if args.dataset == "toolbench":
        ds = ToolbenchDataset()
    elif args.dataset == "apigen":
        ds = APIGenDataset()
    elif args.dataset == "metatool":
        ds = MetaToolDataset()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    # save evaluation results
    eval_result_file_path = os.path.join(args.result_dir, f"{args.dataset}_api{args.tool_top_k}_eval.json")
    if os.path.exists(eval_result_file_path):
        print(f"Loading existing evaluation results from {eval_result_file_path}")
        with open(eval_result_file_path, "r") as f:
            eval_result_dict = json.load(f)
    else:
        eval_result_dict = {}
    eval_result_dict = {int(k): v for k, v in eval_result_dict.items()}

    if args.n_threads == 1:
        # without multithreading for debugging
        for qid, result in result_dict.items():
            evaluate(eval_result_dict, args, qid, ds, result)
    else:
        # multi-threaded evaluation
        with CustomThreadPoolExecutor(max_workers=args.n_threads) as executor:
            for qid, result in result_dict.items():
                executor.submit(evaluate, eval_result_dict, args, qid, ds, result)
    
    # save evaluation results
    with open(eval_result_file_path, "w") as f:
        json.dump(eval_result_dict, f)
    print(f"Saved evaluation results to {eval_result_file_path}")

    # print eval dict stats
    print(f"Evaluation Results ======================")
    print(f"Retrieved {len(eval_result_dict)} results")
    # retrieval
    retrieval_tp = sum([result["retrieval_tp"] for result in eval_result_dict.values()])
    retrieval_fn = sum([result["retrieval_fn"] for result in eval_result_dict.values()])
    retrieval_recall = retrieval_tp / (retrieval_tp + retrieval_fn) if retrieval_tp + retrieval_fn > 0 else 0
    print(f"Retrieval recall: {retrieval_recall}")

    # tool call (whether the correct tool is called)
    tool_tp = sum([result["tool_tp"] for result in eval_result_dict.values()])
    tool_fn = sum([result["tool_fn"] for result in eval_result_dict.values()])
    tool_recall = tool_tp / (tool_tp + tool_fn) if tool_tp + tool_fn > 0 else 0
    print(f"Tool call recall: {tool_recall}")

    # tool call accuracy (whether the correct parameters are called)
    tool_call_correct_count = sum([result["tool_call_correct_count"] for result in eval_result_dict.values()])
    tool_call_count = sum([result["tool_gt_count"] for result in eval_result_dict.values()])
    tool_call_accuracy = tool_call_correct_count / tool_call_count if tool_call_count > 0 else 0
    print(f"Tool call accuracy: {tool_call_accuracy}")

    # solve
    answer_status = [result["answer_status"] for result in eval_result_dict.values()]
    pass_rate = Counter(answer_status)[AnswerStatus.Solved.value] / len(answer_status)
    print(f"Answer status: {Counter(answer_status)}")
    print(f"Pass rate: {pass_rate}")
    # breakpoint()
