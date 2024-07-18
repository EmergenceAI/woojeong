import json
import inspect
from agent_system.src.tool_simulator import APISimulator

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
    # let's ignore type since it's a hassle to convert to python types
    
    func_header = f"def {func_name}("
    param_defaults = []
    
    for param_name, param_info in parameters.items():
        param_type = param_info['type']
        pure_type = param_type.split(',')[0]
        if 'optional' in param_type:
            param_type = param_type.replace(', optional', '')
            param_default = param_info['default'] if param_info['default'] != '' else 'None'
            param_defaults.append(f"{param_name}: {pure_type}={param_default}")
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
    
    exec_globals = {}
    exec(full_function, globals(), exec_globals)

    return exec_globals[func_name]