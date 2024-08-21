import json
import re
import inspect
from agent_system.src.tool_simulator import APISimulator
from typing import List, Dict, Any, Tuple, Union, Optional


def terminate(msg):
    return msg.get("content", "") and msg.get("content", "").rstrip().upper().endswith(
        "##TERMINATE##"
    )


# utils imported from toolbench
def change_name(name):
    if len(name) == 0:
        return name
    if name in ["from", "class", "return", "false", "true", "id", "and", "pass"]:
        name = "is_" + name
    if name in ["type", "print", "input", "open", "file", "exec", "eval", "class", "def"]:
        name = "get_" + name
    if name[0].isdigit():
        name = "get_" + name
    return name


def standardize_category(category):
    save_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in save_category or "," in save_category:
        save_category = save_category.replace(" ", "_").replace(",", "_")
    save_category = save_category.replace("__", "_")
    return save_category


def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+", "_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string


def convert_api_name(tool_name, api_name):
    # print(f"converting api name: {api_name}, tool name: {tool_name}")
    api_name = change_name(standardize(api_name))
    tool_name = standardize(tool_name)
    name = api_name + f"_for_{tool_name}"
    name = standardize(name[-60:])
    return name


def convert_to_openai_tool_schema(func_info):
    """
    Convert apigen function information to OpenAI tool schema.
    """
    schema = {
        "description": func_info["description"],
        "name": func_info["name"],
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    python_to_json_type_mapping = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "set": "array",
        "dict": "object",
    }

    for param_name, param_info in func_info["parameters"].items():
        param_type = param_info["type"]
        pure_param_type = param_type.split(",")[0].strip()
        if pure_param_type not in python_to_json_type_mapping:
            # convert the above exceptions
            if pure_param_type[:4] == "List":
                pure_param_type = "array"
            elif "Tuple" in pure_param_type:
                pure_param_type = "array"
            elif "Dict" in pure_param_type:
                pure_param_type = "object"
            else:
                pure_param_type = "object"
                # raise ValueError(f'Unknown type: {pure_param_type}')
        else:
            pure_param_type = python_to_json_type_mapping[pure_param_type]
        # parse optional
        if "optional" in param_type:
            param_type = pure_param_type
            is_required = False
        else:
            param_type = pure_param_type
            is_required = True

        schema["parameters"]["properties"][param_name] = {
            "type": param_type,
            "description": param_info["description"],
        }

        if is_required:
            schema["parameters"]["required"].append(param_name)

    return {"type": "function", "function": schema}


def pythonize(value, type_str):
    if type_str == "str":
        if type_str == "":
            return '""'
        escaped_string = str(type_str).replace('"', '\\"')
        value = f'"{escaped_string}"'
    elif type_str == "bool":
        return "True" if value in ["true", "True", "1"] else "False"
    elif type_str == "int":
        try:
            value = int(value)
        except:
            value = "None"
    elif type_str == "float":
        try:
            value = float(value)
        except:
            value = "None"
    elif type_str == "list":
        try:
            value = json.loads(value)
        except:
            value = "[]"
    elif type_str == "set":
        try:
            value = set(json.loads(value))
        except:
            value = "set()"
    elif type_str == "dict":
        try:
            value = json.loads(value)
        except:
            value = "{}"
    else:
        value = "None"
    return value


def convert_to_valid_python_type(type_str):
    map_type = {
        "STRING": "str",
        "int": "int",
        "BINARY": "bool",
        "TIME (24-hour HH:MM)": "str",
        "GEOPOINT (latitude, longitude)": "str",
        "str": "str",
        "ENUM": "str",
        "BOOLEAN": "bool",
        "DATE (YYYY-MM-DD)": "str",
        "FILE": "str",
        "NUMBER": "float",
        "string": "str",
        "STRING": "str",
        "OBJECT": "str",
        "ARRAY": "list",
        # None,
    }
    if type_str is None:
        return "object"
    if type_str in map_type:
        type_str = map_type[type_str]
        return type_str
    # check pure_type is valid
    if type_str not in ["str", "int", "float", "bool", "list", "set", "dict"]:
        if type_str.startswith("List"):
            type_str = "list"
        elif type_str.startswith("Tuple"):
            type_str = "tuple"
        elif type_str.startswith("Dict"):
            type_str = "dict"
        else:
            type_str = "object"
            # raise ValueError(f'Unknown type: {type_str}')
    return type_str


def convert_api_for_registration(dataset, api, max_desc_len=1000):
    """
    Convert the API to a standard format for function registration.
    Example:
    {
        'name': 'light_travel_time',
        'description': 'Calculate the time taken for light to travel from one celestial body to another.',
        'required_params': [
            {
                'name': 'distance_in_light_years',
                'description': 'The distance between the two celestial bodies in light years.',
                'type': 'int'
            }
        ],
        'optional_params': [
            {
                'name': 'speed_of_light',
                'description': 'The speed of light in vacuum, in m/s. Default value is 299792458 m/s.',
                'type': 'int',
                'default': 299792458
            }
        ]
    }
    """
    if dataset == "toolbench":
        required_params, optional_params = [], []
        for param in api["api_required_parameters"]:
            # print(f"converting param: {param['name']}")
            param_name = change_name(standardize(param["name"]))
            pure_type = convert_to_valid_python_type(param["type"])
            # if param is duplicated, pass
            if param_name in [p["name"] for p in required_params]:
                continue
            required_params.append(
                {
                    "name": param_name,
                    "description": param["description"],
                    "type": pure_type,
                }
            )
        for param in api["api_optional_parameters"]:
            param_name = change_name(standardize(param["name"]))
            pure_type = convert_to_valid_python_type(param["type"])
            default_value = pythonize(param.get("default", "None"), pure_type)
            # if param is duplicated, pass
            if param_name in [p["name"] for p in required_params + optional_params]:
                continue
            optional_params.append(
                {
                    "name": param_name,
                    "description": param["description"],
                    "type": pure_type,
                    "default": default_value,
                }
            )

        # description can't be empty
        description = api.get("api_description", "")
        description = description[:max_desc_len]
        if description.strip() == "":
            description = f"This is the subfunction for tool \"{api['tool_name']}\", you can use this tool."

        converted_api = {
            "name": convert_api_name(api["tool_name"], api["api_name"]),
            "description": description,
            "required_params": required_params,
            "optional_params": optional_params,
        }
    elif dataset == "apigen":
        required_params = []
        optional_params = []
        for param_name, param_info in api["parameters"].items():
            param_type = param_info["type"]
            # # there exist something like List[Union[int, float]]
            pure_type = convert_to_valid_python_type(param_type.split(",")[0].strip())
            if "optional" in param_type:
                param_default = param_info.get("default", "None")
                param_default = pythonize(param_default, pure_type)
                optional_params.append(
                    {
                        "name": param_name,
                        "description": param_info["description"],
                        "type": pure_type,
                        "default": param_default,
                    }
                )
            else:
                required_params.append(
                    {
                        "name": param_name,
                        "description": param_info["description"],
                        "type": pure_type,
                    }
                )

        api["required_params"] = required_params
        api["optional_params"] = optional_params
        converted_api = api
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    return converted_api


def convert_api_to_function(api):
    """
    Convert the API to a function that can be registered as a tool.
    Function example:
    def get_post_comments(pid: str, count: str, cursor: str=None):
        Fetch comments for a given Twitter post using its post ID.
        :param: The post ID for which to retrieve comments.
        :param: The number of comments to retrieve.
        :param: Cursor for pagination to retrieve the next set of comments. Defaults to None.

        simulator = APISimulator({"name": "get_post_comments", "description": "Fetch comments for a given Twitter post using its post ID.", "parameters": {"pid": {"description": "The post ID for which to retrieve comments.", "type": "str", "default": "1552735248026411010"}, "count": {"description": "The number of comments to retrieve.", "type": "str", "default": "40"}, "cursor": {"description": "Cursor for pagination to retrieve the next set of comments. Defaults to None.", "type": "str, optional", "default": ""}}})
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        api_input = {arg: values[arg] for arg in args}
        return simulator.run(api_input)

    Args:
        api (dict): The API dictionary.

    Returns:
        function: The function that can be registered as a tool.

    """
    func_name = api["name"]
    description = api["description"]
    required_params = api["required_params"]
    optional_params = api["optional_params"]

    func_header = f"def {func_name}("
    param_defaults = []

    for param in required_params:
        param_defaults.append(f"{param['name']}: {param['type']}")
    for param in optional_params:
        param_defaults.append(f"{param['name']}: {param['type']}={param['default']}")

    func_defaults = ", ".join(param_defaults)

    func_header += func_defaults + "):"

    docstring = f'    """\n    {description}\n'
    # # add parameter descriptions
    # for param in required_params + optional_params:
    #     docstring += f'    :param {param["name"]}: {param["description"]}\n'

    docstring += '    """\n'

    func_body = f"""
    simulator = APISimulator({json.dumps(api)})
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    api_input = {{arg: values[arg] for arg in args}}
    return simulator.run(api_input) 
    """

    full_function = f"{func_header}\n{docstring}{func_body}"
    # breakpoint()

    exec_globals = {}
    try:
        exec(full_function, globals(), exec_globals)
    except Exception as e:
        print(f"Error while executing function: {e}")
        print(full_function)
        breakpoint()

    return exec_globals[func_name]


def compare_tool_calls(gt, pred):
    if gt["name"] != pred["name"]:
        return False
    # convert string to dictionary
    if type(pred["arguments"]) == str:
        pred["arguments"] = json.loads(pred["arguments"])
    if type(gt["arguments"]) == str:
        gt["arguments"] = json.loads(gt["arguments"])

    if len(gt["arguments"]) != len(pred["arguments"]):
        return False
    for k in gt["arguments"]:
        if k not in pred["arguments"]:
            return False
        if gt["arguments"][k] != pred["arguments"][k]:
            return False
    return True
