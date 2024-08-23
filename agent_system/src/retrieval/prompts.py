TOOL_SUMMARY_PROMPT = """I will provide you information about an api, including the list of endpoints that constitute the api. I will provide this information as a json object. 
Please produce a natural language summary describing the API's capabilities. Your summary should be around 200 words. 
If you're familiar with the underlying web-service (like an Amazon API or weather.com API), briefly describe it.
Your description should enable someone who doesn't know about the API to understand when they should use the API."""

API_SUMMARY_PROMPT = """I will provide you information about an api, including the category, the tool name and description of the api belongs to, and the information of that specific api including name, description, and parameters. I will provide this information as a json object. 
Please produce a natural language summary describing the API's capabilities. Your summary should be around 200 words. 
If you're familiar with the underlying web-service (like an Amazon API or weather.com API), briefly describe it.
Your description should enable someone who doesn't know about the API to understand when they should use the API."""

# NOTE make it shorter or focus on the user intent rather than api description
API_INTENT_PROMPT = """I will provide you information about an API, including the name, description, and input parameters. 
There are two types of parameters: required and optional. Required parameters are necessary for the API to function, while optional parameters are not.
You can figure out this information by looking at each parameter's type. I will provide this information in a json format. 
There APIs can be used to solve queries given by users.

Please produce a natural language summary describing the API's capabilities. Your summary should be around 200 words. 
If you're familiar with the underlying web-service (like an Amazon API or weather.com API), briefly describe it.
Your description should enable someone who doesn't know about the API to understand when they should use the API.
Please provide a brief description of the API's intended use case.
"""
