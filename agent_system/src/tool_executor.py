import os
import requests
import autogen
import logging
import argparse
import time
import pickle
import json
from dotenv import load_dotenv
from tqdm import tqdm
from enum import Enum
from agent_system.src.utils import terminate, standardize
from agent_system.src.prompts import LLM_PROMPTS
from toolbench_analysis.src.load_toolbench_utils import load_api_data, load_query_api_mapping, load_query_data


class APIStatus(Enum):
    SUCCESS = 0
    API_NOT_WORKING = 6
    UNAUTHORIZED = 7
    UNSUBSCRIBED = 8
    TOO_MANY_REQUESTS = 9
    RATE_LIMIT_PER_MINUTE = 10
    MESSAGE_ERROR = 11
    INVALID_DATA = 12
    TOOLBENCH_SERVER_ERROR = 13


class APIExecutor():
    def __init__(self):
        self.service_url = "http://8.218.239.54:8080/rapidapi"
        self.toolbench_key = os.getenv("TOOLBENCH_API_KEY")
        self.headers = {
            "toolbench_key": self.toolbench_key,
        }

    def parse_response(self, response):
        """Parse response from RapidAPI server
        Args:
            response (requests.Response): response from RapidAPI server
        
        Returns:
            str: response in json format
            int: status code
        """
        try:
            response = response.json()
        except:
            print(response)
            return json.dumps({"error": f"request invalid, data error", "response": ""}), 12
        
        if response["error"] == "API not working error...":
            status_code = 6
        elif response["error"] == "Unauthorized error...":
            status_code = 7
        elif response["error"] == "Unsubscribed error...":
            status_code = 8
        elif response["error"] == "Too many requests error...":
            status_code = 9
        elif response["error"] == "Rate limit per minute error...":
            print("Reach api calling limit per minute, sleeping...")
            time.sleep(10)
            status_code = 10
        elif response["error"] == "Message error...":
            status_code = 11
        else:
            status_code = 0
        return json.dumps(response), status_code
    
    def run(self, api_data: dict):
        """Run API call
        Args:
            api_data (dict): dictionary containing the api data
        
        Returns:
            str: response in json format
            int: status code
        """
        payload = {
            "category": api_data["category_name"],
            "tool_name": api_data["tool_name"],
            "api_name": api_data["api_name"],
            "tool_input": api_data["input_parameters"],
            "strip": "truncate",
            "toolbench_key": self.toolbench_key,
        }
        try:
            response = requests.post(
                self.service_url,
                headers=self.headers,
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
        except requests.exceptions.ReadTimeout as err:
            return json.dumps({"error": f"error occurred while calling Toolbench server, check if valid api is given", "response": ""}), 13
        except requests.exceptions.HTTPError as err:
            return json.dumps({"error": f"error occurred while calling Toolbench server, check if valid api is given", "response": ""}), 13
        except:
            return json.dumps({"error": f"error occurred while calling Toolbench server, check if valid api is given", "response": ""}), 13
        
        response, status_code = self.parse_response(response)
        return response, status_code
    
def call_api_executor(api_data: dict) -> tuple[dict, int]:
    api_executor = APIExecutor()
    response, status_code = api_executor.run(
        api_data["category_name"],
        api_data["tool_name"],
        api_data["api_name"],
        api_data["input_parameters"],
    )
    return response, status_code


def create_function_code(func_info):
    # Extract information from the dictionary
    func_name = func_info['func_name']
    required_params = func_info['api_required_parameters']
    optional_params = func_info['api_optional_parameters']


    # required_params consist of the list of dictionaries
    # there is "type" field in each dictionary
    # but this is 'STRING' or 'INT' -> should be 'str' or 'int'
    # convert 'STRING' to 'str' and 'INT' to 'int'
    string_to_type = {
        "STRING": "str",
        'int': 'int',
        'BINARY': 'bool',
        'TIME (24-hour HH:MM)': 'str',
        'GEOPOINT (latitude, longitude)': 'str',
        'str': 'str',
        'ENUM': 'str',
        'BOOLEAN': 'bool',
        'DATE (YYYY-MM-DD)': 'str',
        'FILE': 'str',
        'NUMBER': 'float',
        'string': 'str',
        'STRING': 'str',
        'OBJECT': 'str',
        'ARRAY': 'list',
        # None,
    }
    default_values = {
        'str': '""',
        'int': '0',
        'bool': 'False',
    }

    for param in required_params:
        param["type"] = string_to_type[param["type"]]
    for param in optional_params:
        param["type"] = string_to_type[param["type"]]
    
    # Generate the parameter string with type annotations
    # add double quotes to the default value if it is a string
    for param in required_params:
        param["name"] = standardize(param["name"])
        if param["type"] == "str":
            param["default"] = f'"{param["default"]}"'
    for param in optional_params:
        param["name"] = standardize(param["name"])
        if param["type"] == "str":
            param["default"] = f'"{param["default"]}"'
    
    # set default values for optional parameters, if not provided
    for param in optional_params:
        if param["default"] in [None, "None", '']:
            param["default"] = default_values[param["type"]]
    
    # Generate the parameter string with type annotations
    param_str = ", ".join([f"{param['name']}: {param['type']}" for param in required_params])
    if optional_params:
        optional_str = ", ".join([f"{param['name']}: {param['type']} = {param['default']}" for param in optional_params])
        param_str = f"{param_str}, {optional_str}" if param_str else optional_str
    
    # Generate the function code with type annotations
    function_code = f"""
def {func_name}({param_str}):
    # {func_info["category_name"]}, {func_info["tool_name"]}, {func_info["api_name"]}
    print('Function {func_name} called with:', {', '.join([param['name'] for param in required_params] + [param['name'] for param in optional_params])})
    # execute api
    api_executor = APIExecutor()
    response, status_code = api_executor.run(api_data)
    return response, status_code
"""
    return function_code

def convert_apis_to_functions(apis, globals=globals()):
    for api in tqdm(apis):
        # get "api_name" or "name" from api
        api_name = api.get("api_name", api.get("name"))
        func_name = standardize(api_name)
        api['func_name'] = func_name
        # query the api
        func_code = create_function_code(api)
        print(func_code)
        exec(func_code, globals)
        # except Exception as e:
        #     print(f"Error: {e}")
        #     breakpoint()
        # Store the function in the dictionary
        api["func"] = globals[func_name]
        print(f"Function '{func_name}' created successfully")


def main(args):
    # load datasets
    query_api_mapping, id2doc, id2query = load_query_api_mapping(local_file_path="../toolbench_analysis/data/filtered_query_api_mapping.csv")
    api_data = load_api_data()
    # query_data = load_query_data("g1")
    with open("../toolbench_analysis/data/docid2api.pkl", "rb") as f:
        docid2api = pickle.load(f)

    # merge api_data with query_api_mapping
    api_data.reset_index(inplace=True)
    api_data = api_data.rename(columns={"index": "api"}).set_index("api")

    # convert apis to functions
    api_dict = api_data.to_dict(orient="records")
    convert_apis_to_functions(api_dict)

    # join api_data_subset to query_api_mapping
    query_api_mapping["api"] = query_api_mapping["docid"].apply(lambda x: docid2api[x])
    query_api_mapping = query_api_mapping.join(api_data, on="api")

    # select 1 query for testing
    qid = 570
    # 489
    data = query_api_mapping[query_api_mapping["qid"] == qid]
    data = data.to_dict(orient="records")

    query = data[0]["query"]
    n_apis = len(data)
    print(f"Query: {query}")
    print(f"Number of APIs: {n_apis}")

    # === APIs
    # first let's define api tools
    api = data[0]
    category_name = api["category_name"]
    tool_name = api["tool_name"]
    api_name = api["api_name"]
    tool_description = api["tool_description"]
    api_description = api["api_description"]
    required_parameters = api["api_required_parameters"]
    optional_parameters = api["api_optional_parameters"]

    # generate dummy parameters
    input_parameters = {}
    for param in required_parameters:
        input_parameters[param["name"]] = param["default"]
    for param in optional_parameters:
        input_parameters[param["name"]] = param["default"]
    api["input_parameters"] = json.dumps(input_parameters)

    direct_call = True
    if direct_call:
        # call the api executor
        response, status_code = call_api_executor(api)
    else:
        # Generate and execute the function code
        # NOTE this is very inefficient, ideally calling api.py should do its job
        for func_info in data:
            func_name = standardize(func_info['api_name'])
            func_info['func_name'] = func_name
            func_code = create_function_code(func_info)
            exec(func_code)
            # Store the function in the dictionary
            func_info["func"] = locals()[func_name]
    
    # unique_string_types = []
    # for param_list in api_data['api_required_parameters'].values:
    #     for param in param_list:
    #         unique_string_types.append(param['type'])
    # for param_list in api_data['api_optional_parameters'].values:
    #     for param in param_list:
    #         unique_string_types.append(param['type'])
    # unique_string_types = set(unique_string_types)

    # ========= Autogen
    # config llm
    llm_config={
        "config_list": [
            {
                "model": os.getenv("AUTOGEN_MODEL_NAME"),
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        ],
        "cache_seed": None
    }

    # Create an user proxy
    user = autogen.UserProxyAgent(
        name="user",
        is_termination_msg=terminate,
        human_input_mode="NEVER",
        llm_config=None,
        max_consecutive_auto_reply=args.number_of_rounds,
        code_execution_config={
            "last_n_messages": 1,
            # "work_dir": "tasks",
            "use_docker": False,
        },
    )

    orchestrator = autogen.AssistantAgent(
        name="orchestrator",
        system_message=LLM_PROMPTS["ORCHESTRATOR_PROMPT"],
        llm_config=llm_config,
        is_termination_msg=terminate,
    )

    for api_data in data:
        print(api_data["api_name"])
        print(api_data["func_name"])
        autogen.register_function(
            api_data["func"],
            caller=orchestrator,  # The assistant agent can suggest calls to the calculator.
            executor=user,  # The user proxy agent can execute the calculator calls.
            name=api_data["func_name"],  # By default, the function name is used as the tool name.
            description=api_data["api_description"],  # A description of the tool.
        )

        # inspect function params
        # import inspect
        # print(inspect.signature(api_data["func"]))
    print(orchestrator.llm_config["tools"])

    res = user.initiate_chat(
        orchestrator,
        message=query,
        summary_method="last_msg",
    )
    breakpoint()


if __name__ == "__main__":
    load_dotenv(".env")
    args = parse_args()
    main(args)