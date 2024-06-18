import json
import logging
import os

import openai

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)

def store_api_data(api_dict, output_path):
    """Stores the API data in a JSON file.

    Args:
        api_dict (dict): The dictionary containing the API data.
        output_path (str): The path to the output JSON file.
    """
    with open(output_path, 'w') as json_file:
        json.dump(api_dict, json_file)

def read_api_data(input_path):
    """Reads the API data from a JSON file.

    Args:
        input_path (str): The path to the input JSON file.

    Returns:
        dict: A dictionary containing the API data.
    """
    try:
        with open(input_path, 'r') as json_file:
            return json.load(json_file)
    except Exception as e:
        logging.error(f"Error reading API data: {e}")
        return None
    
def get_gpt_response(
    messages,
    model="gpt-4-turbo-preview",
    temperature=0
):
    logging.info("Getting gpt response ...")
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=False
    )
    logging.info(f"GPT response: {response}")
    try:
        output = response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error getting gpt response: {e}")
        output = ""
    return output
