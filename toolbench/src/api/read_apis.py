import os
import json
import logging
from dotenv import load_dotenv
from src.api.utils import store_api_data

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)

def read_apis_from_folder(folder_path):
    """Reads APIs from a folder and returns a dictionary of APIs.
    
    Args:
        folder_path (str): The path to the subfolder containing the APIs.
        
    Returns:
        dict: A dictionary containing the API data.
    """
    api_dict = {}

    # Iterate over all files in the subdirectory
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        
        # Check if the file is a JSON file
        if os.path.isfile(file_path) and file.endswith('.json'):
            
            api_name = file.split('.')[0]

            # Chek if a subfolder exists that has the same name as the json file
            # This subfolder should have an api.py file (which provides the API invocation logic)
            subfolder_path = os.path.join(folder_path, api_name)
            if os.path.isdir(subfolder_path) and os.path.exists(os.path.join(subfolder_path, 'api.py')):

                # Open and parse the JSON file
                with open(file_path, 'r') as json_file:
                    api_dict[api_name] = json.load(json_file)
        
        logging.info(f"Processed API {api_name}")

    return api_dict

def read_apis(path):
    """Reads APIs from subfolders of a folder and returns a dictionary of APIs.

    Args:
        path (str): The path to the folder containing the APIs.

    Returns:
        dict: A dictionary containing the API data.
    """
    api_dict = {}
    for entry in os.listdir(path):
        subfolder_path = os.path.join(path, entry)
        
        # Check if this is a directory
        if os.path.isdir(subfolder_path):
            try:
                api_dict[entry] = read_apis_from_folder(subfolder_path)
            except Exception as e:
                logging.error(f"Error processing subfolder {subfolder_path}: {e}")
        
        logging.info(f"Processed subfolder {entry}")

    return api_dict

def main():
    data_folder = os.path.join(os.getenv("TOOLBENCH_FOLDER"), "data", "toolenv", "tools")
    api_data = read_apis(data_folder)
    store_api_data(api_data, os.getenv("API_DATA_STORE"))

if __name__ == "__main__":
    load_dotenv(".env")
    main()
