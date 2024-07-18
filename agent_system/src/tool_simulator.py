import json
import time

SIMULATOR_PROMPT = '''
    Imagine you are an API Server. Your role is to simulate API calls based on the API documentation provided in a JSON format. API documentation includes the API's name, description, and input parameters. There are two types of parameters: required and optional. Optional parameters are specified as "optional" in the "type" field.

    Following is the documentation for the API you need to simulate:
        
        {API_INFO}

    Your task is to generate a JSON response that aligns with the expected output of the API. As you receive specific inputs for this API call, analyze these inputs to determine their intended purpose.
    Your responses must adhere to a specific JSON structure, which is as follows:\n
    {
        "error": "",
        "response": "<Your_Response>"
    }\n
    The error field should remain empty, indicating no errors in processing. The response field should contain the content you formulate based on the API's functionality and the input provided. Ensure that your responses are meaningful, directly addressing the API's intended functionality. If the provided examples are mostly error messages or lack substantial content, use your judgment to create relevant and accurate responses. The key is to maintain the JSON format's integrity while ensuring that your response is an accurate reflection of the API's intended output.\n
    Please note that your answer should not contain anything other than a json format object, which should be parsable directly to json.
    Note that:
    - your response should be around 100 to 200 words, containing rich information given the api input parameters. Keep Your answer short and simple.
    - your response must be effective and have practical content.
    - do not mention that this is a simulation in your response, assume that this is a real scenario and provide imaginary responses if the information required is not available
'''


# SIMULATOR_PROMPT = '''
#     Imagine you are an API Server operating within a specialized tool, which contains a collection of distinct APIs. Your role is to deeply understand the function of each API based on their descriptions in the API documentation. As you receive specific inputs for individual API calls within this tool, analyze these inputs to determine their intended purpose. Your task is to craft a JSON formatted response that aligns with the expected output of the API, guided by the provided examples.\n
#     Your responses must adhere to a specific JSON structure, which is as follows:\n
#     {
#         "error": "",
#         "response": "<Your_Response>"
#     }\n
#     The error field should remain empty, indicating no errors in processing. The response field should contain the content you formulate based on the API's functionality and the input provided. Ensure that your responses are meaningful, directly addressing the API's intended functionality. If the provided examples are mostly error messages or lack substantial content, use your judgment to create relevant and accurate responses. The key is to maintain the JSON format's integrity while ensuring that your response is an accurate reflection of the API's intended output within the tool.\n
#     Please note that your answer should not contain anything other than a json format object, which should be parsable directly to json.
#     Note that:
#     - your response should be around 100 to 200 words, containing rich information given the api input parameters. Keep Your answer short and simple.
#     - your response must be effective and have practical content.
#     - if the api response example if null or ineffective, ignore the example and give your independent response.
# '''

class APISimulator():
    """
    Simulate API calls with gpt-4-0125-preview model
    """
    def __init__(self, api_data: dict, model_name: str = "gpt-4-0125-preview"):
        self.api_data = api_data
        self.model_name = model_name
        self.system_prompt = self.build_system_prompt(api_data)

    def build_system_prompt(self, api_data):
        """Build system prompt for API call simulation
        
        Returns:
            dict: system prompt
        """
        system_prompt = {
            "role": "system",
            "content": SIMULATOR_PROMPT.replace("{API_INFO}", json.dumps(api_data)),
        }
        return system_prompt


    def parse_response(self, response):
        """Parse response from RapidAPI server
        Args:
            response (requests.Response): response from RapidAPI server
        
        Returns:
            str: response in json format
            int: status code
        """
        try:
            response = json.loads(response)
        except:
            print(response)
            return json.dumps({"error": f"JSON parsing error", "response": ""}), 12

        if "error" not in response or "response" not in response:
            return json.dumps({"error": "Simulator output format error", "response": ""}), 13
        
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
        return json.dumps(response)
    
    def run(self, api_input: dict):
        """Run API call
        Args:
            api_input (dict): dictionary containing the input for the API call
        
        Returns:
            str: response in json format
            int: status code
        """
        # user prompt, truncated to 2048 characters if too long
        user_prompt = "API Input: " + json.dumps(api_input)
        user_prompt = {"role": "user", "content": user_prompt}

        from toolbench_analysis.src.api.utils import get_gpt_response
        response = get_gpt_response(
            messages=[self.system_prompt, user_prompt], 
            model=self.model_name,
        )
        return self.parse_response(response)