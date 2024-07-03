import json
import os
from dotenv import load_dotenv
import pandas as pd
import logging
from toolbench_analysis.src.api.utils import store_api_data

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)

# filter APIs by score key and score threshold
def filter_apis_by_score(api_dict, score_key, threshold):
    """ Filter APIs by score key and score threshold
    Args:
        api_dict (dict): dictionary of APIs
        score_key (str): key to filter by
        threshold (int): threshold value
    Returns:
        dict: filtered dictionary of APIs
    """
    filtered_dict = {}
    for api_class in api_dict.keys():
        filtered_dict[api_class] = {}
        for api in api_dict[api_class].keys():
            api_score = api_dict[api_class][api].get("score", None)
            if api_score is not None and api_score.get(score_key, 0) >= threshold:
                filtered_dict[api_class][api] = api_dict[api_class][api]
    return filtered_dict

# compute api statistics: api class, api, score keys and values
# return a pandas dataframe
def compute_api_statistics(api_dict):
    """ Compute API dictionary statistics
    Args:
        api_dict (dict): dictionary of APIs
    Returns:
        pd.DataFrame: pandas dataframe of API statistics
    """
    api_stats = []
    for api_class in api_dict:
        for api in api_dict[api_class]:
            api_score = api_dict[api_class][api].get("score", None)
            if api_score is not None:
                api_stats.append({
                    "api_class": api_class,
                    "api": api,
                } | api_score)
    api_stats_df = pd.DataFrame(api_stats)
    logging.info(f"API statistics:\n{api_stats_df.describe()}")
    return api_stats_df
                                  
# compute api class statistics: api class, number of apis and number of endpoints
# return a pandas dataframe
def compute_api_class_statistics(api_dict):
    """ Compute API class statistics
    Args:
        api_dict (dict): dictionary of APIs
    Returns:    
        pd.DataFrame: pandas dataframe of API class statistics
    """
    api_class_stats = []
    for api_class in api_dict:
        num_apis = len(api_dict[api_class])
        num_endpoints = sum([len(api_dict[api_class][api]["api_list"]) for api in api_dict[api_class]])
        api_class_stats.append({
            "api_class": api_class,
            "num_apis": num_apis,
            "num_endpoints": num_endpoints
        })
    api_class_stats_df = pd.DataFrame(api_class_stats)
    logging.info(f"API class statistics:\n{api_class_stats_df}")
    return api_class_stats_df

def compute_api_data_statistics(api_dict):
    api_class_stats_df = compute_api_class_statistics(api_dict)
    api_stats_df = compute_api_statistics(api_dict)
    logging.info(f"Total APIs: {api_class_stats_df['num_apis'].sum()}")
    logging.info(f"Total API endpoints: {api_class_stats_df['num_endpoints'].sum()}")    
    return api_class_stats_df, api_stats_df

def main(
        score_key="popularityScore",
        threshold=9.5,
):
    with open(os.getenv("API_DATA_STORE"), 'r') as json_file:
        api_dict = json.load(json_file)
    logging.info("Base API data stats:")
    compute_api_class_statistics(api_dict)
    
    filtered_api_dict = filter_apis_by_score(api_dict, score_key, threshold)
    logging.info("Filtered API data stats:")
    compute_api_class_statistics(filtered_api_dict)

    output_file = os.getenv("FILTERED_API_DATA_STORE_PREFIX") + "_" + score_key + "_" + str(threshold) + ".json"
    store_api_data(filtered_api_dict, output_file)

if __name__ == "__main__":
    load_dotenv(".env")
    main()