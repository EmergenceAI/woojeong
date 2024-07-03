TOOL_SUMMARY_PROMPT = """I will provide you information about an api, including the list of endpoints that constitute the api. I will provide this information as a json object. 
Please produce a natural language summary describing the API's capabilities. Your summary should be around 200 words. 
If you're familiar with the underlying web-service (like an Amazon API or weather.com API), briefly describe it.
Your description should enable someone who doesn't know about the API to understand when they should use the API."""

API_SUMMARY_PROMPT = """I will provide you information about an api, including the category, the tool name and description of the api belongs to, and the information of that specific api including name, description, and parameters. I will provide this information as a json object. 
Please produce a natural language summary describing the API's capabilities. Your summary should be around 200 words. 
If you're familiar with the underlying web-service (like an Amazon API or weather.com API), briefly describe it.
Your description should enable someone who doesn't know about the API to understand when they should use the API."""
