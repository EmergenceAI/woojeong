import os
import pickle
from dotenv import load_dotenv

from agent_system.src.tool_datasets import ToolbenchDataset, APIGenDataset
from agent_system.src.autogen_wrapper import AutogenWrapper
from agent_system.src.tool_executor import convert_apis_to_functions


if __name__ == "__main__":
    load_dotenv(".env")
    qid = 1  # 530

    # ds = ToolbenchDataset(filter_query_api_mapping=True)
    ds = APIGenDataset()
    query = ds.get_query_by_id(qid)
    apis = ds.get_apis_by_query_id(qid)
    answers = ds.get_answers_by_query_id(qid)

    print(f"Query: {query}")
    print(f"Number of APIs: {len(apis)}")
    print(f"Number of API calls: {len(answers)}")

    # convert apis to function and register them
    # make sure the functions are available in the local scope
    convert_apis_to_functions(apis, globals())
    # print("global functions: ", globals().keys())
    # breakpoint()

    # ==== instantiate autogen wrapper
    autogen_wrapper = AutogenWrapper(max_chat_round=50)
    autogen_wrapper.create(["user", "orchestrator", "tool_executor", "tool_execution_manager"])
    print("AutogenWrapper created successfully")

    # register functions
    autogen_wrapper.register_tools(apis)
    
    autogen_wrapper.initiate_chat(user_query=query)