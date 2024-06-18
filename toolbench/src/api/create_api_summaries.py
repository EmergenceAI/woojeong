import json
import os
from dotenv import load_dotenv

from src.api.utils import get_gpt_response, read_api_data, store_api_data
from src.api.prompts import API_SUMMARY_PROMPT

def _create_summary_prompt(
    api_description
):
    """Create prompt for LLM to generate API summaries for future retrieval.

    Args:
        api_description (str): API description to summarize

    Returns:
        str: API summary prompt
    """
    messages = []
    messages.append({"role": "system", "content": API_SUMMARY_PROMPT})
    messages.append({"role": "user", "content": api_description})
    return messages

def create_raw_api_description(
        api_data
):
    """Create raw API description from data for a given api.
    This raw description will subsequently be sent to the LLM for summarization.

    Args:
        api_data (dict): Dictionary containing the API data

    Returns:
        str: API description
    """
    #select tool_name, tool_description and title fields from the api_data
    print(api_data)
    api_data_subset = {
        k: api_data[k] for k in (
            'tool_name', 'tool_description', 'title'
        )
    }
    api_data_subset["api_list"] = []
    for endpoint in api_data["api_list"]:
        api_data_subset["api_list"].append({
            k: endpoint[k] for k in (
                "name", 'description',
                'required_parameters',
                'optional_parameters'
            )
        })
    return json.dumps(api_data_subset)

def create_api_summaries(
    api_data_path
):
    """Create API summaries for all APIs in the API data.

    Args:
        api_data_path (str): Path to the API data JSON file

    Returns:
        dict: A dictionary containing the API summaries.
    """
    api_data = read_api_data(api_data_path)
    api_summaries = {}
    for api_class in api_data:
        api_summaries[api_class] = {}
        for api in api_data[api_class]:
            api_description = create_raw_api_description(api_data[api_class][api])
            messages = _create_summary_prompt(api_description)
            api_summaries[api_class][api] = get_gpt_response(messages)
    return api_summaries

def main(
    score_key="popularityScore",
    threshold=9.5,
):
    input_file = os.getenv("FILTERED_API_DATA_STORE_PREFIX") + \
        "_" + score_key + "_" + str(threshold) + ".json"
    api_summaries = create_api_summaries(input_file)
    store_api_data(api_summaries, os.getenv("LLM_GENERATED_API_SUMMARY_STORE"))

if __name__ == "__main__":
    load_dotenv(".env")
    main()