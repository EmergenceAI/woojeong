import json
import inspect
from agent_system.src.tool_simulator import APISimulator
from typing import List, Dict, Any, Tuple, Union, Optional

def terminate(msg):
    return msg.get("content", "") and msg.get("content", "").rstrip().upper().endswith("##TERMINATE##")


def convert_to_openai_tool_schema(func_info):
    """
    Convert apigen function information to OpenAI tool schema.
    """
    schema = {
        'description': func_info['description'],
        'name': func_info['name'],
        'parameters': {
            'type': 'object',
            'properties': {},
            'required': []
        }
    }
    python_to_json_type_mapping = {
        'str': 'string',
        'int': 'integer',
        'float': 'number',
        'bool': 'boolean',
        'list': 'array',
        'set': 'array',
        'dict': 'object'
    }

    for param_name, param_info in func_info['parameters'].items():
        param_type = param_info['type']
        pure_param_type = param_type.split(',')[0].strip()
        if pure_param_type not in python_to_json_type_mapping:
            # convert the above exceptions
            if pure_param_type[:4] == 'List':
                pure_param_type = 'array'
            elif 'Tuple' in pure_param_type:
                pure_param_type = 'array'
            elif 'Dict' in pure_param_type:
                pure_param_type = 'object'
            else:
                pure_param_type = 'object'
                # raise ValueError(f'Unknown type: {pure_param_type}')
        else:
            pure_param_type = python_to_json_type_mapping[pure_param_type]
        # parse optional
        if 'optional' in param_type:
            param_type = pure_param_type
            is_required = False
        else:
            param_type = pure_param_type
            is_required = True

        schema['parameters']['properties'][param_name] = {
            'type': param_type,
            'description': param_info['description']
        }
        
        if is_required:
            schema['parameters']['required'].append(param_name)
    
    return {'type': 'function', 'function': schema}


def pythonize(value, type_str):
    if type_str == 'str':
        value = f'"{value}"'
        value.replace("'", "\'")
    if type_str == 'bool':
        return 'True' if value in ['true', 'True', '1'] else 'False'


def convert_to_valid_python_type(type_str):
    # check pure_type is valid
    if type_str not in ['str', 'int', 'float', 'bool', 'list', 'set', 'dict']:
        if type_str.startswith('List'):
            type_str = 'list'
        elif type_str.startswith('Tuple'):
            type_str = 'tuple'
        elif type_str.startswith('Dict'):
            type_str = 'dict'
        else:
            type_str = 'object'
            # raise ValueError(f'Unknown type: {type_str}')
    return type_str


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
    func_name = api['name']
    description = api['description']
    parameters = api['parameters']
    
    func_header = f"def {func_name}("
    param_defaults = []
    
    optional_flag = False
    for param_name, param_info in parameters.items():
        param_type = param_info['type']
        # # there exist something like List[Union[int, float]]
        pure_type = convert_to_valid_python_type(param_type.split(',')[0].strip())
        if 'optional' in param_type or optional_flag:
            param_type = param_type.replace(', optional', '')
            param_default = param_info.get("default", 'None')
            param_default = pythonize(param_default, pure_type)
            param_defaults.append(f"{param_name}: {pure_type}={param_default}")
            # once we see optional, all the following parameters are optional
            optional_flag = True
        else:
            param_defaults.append(f"{param_name}: {pure_type}")

    func_defaults = ', '.join(param_defaults)
    
    func_header += func_defaults + "):"
    
    docstring = f'    """\n    {description}\n'
    for param_name, param_info in parameters.items():
        docstring += f"    :param: {param_info['description']}\n"
    
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

    return exec_globals[func_name]


def compare_tool_calls(gt, pred):
        if gt['name'] != pred['name']:
            return False
        # convert string to dictionary
        if type(pred['arguments']) == str:
            pred['arguments'] = json.loads(pred['arguments'])
        if type(gt['arguments']) == str:
            gt['arguments'] = json.loads(gt['arguments'])

        if len(gt['arguments']) != len(pred['arguments']):
            return False
        for k in gt['arguments']:
            if k not in pred['arguments']:
                return False
            if gt['arguments'][k] != pred['arguments'][k]:
                return False
        return True