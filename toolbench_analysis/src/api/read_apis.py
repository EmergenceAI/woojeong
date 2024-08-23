import os
import json
import logging
from dotenv import load_dotenv
from toolbench_analysis.src.utils import store_api_data

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

    files = os.listdir(folder_path)
    # get json files only
    json_files = sorted([file for file in files if file.endswith(".json")])
    # TODO: can we use apis with only api.py without json file?

    # Iterate over all files in the subdirectory
    for json_file in json_files:
        api_name = json_file.split(".")[0]
        json_file_path = os.path.join(folder_path, json_file)
        subfolder_path = os.path.join(folder_path, api_name)

        # Check if a subfolder exists that has the same name as the json file
        # This subfolder should have an api.py file (which provides the API invocation logic)
        if os.path.isdir(subfolder_path) and os.path.exists(
            os.path.join(subfolder_path, "api.py")
        ):
            # Open and parse the JSON file
            # NOTE there can be multiple APIs in a single json file
            with open(json_file_path, "r") as json_file:
                api_dict[api_name] = json.load(json_file)

        else:
            logging.error(
                f"Tool {api_name} does not have a corresponding subfolder with an api.py file"
            )
        logging.info(f"Processed Tool {api_name}")

    return api_dict


def read_apis(path):
    """Reads APIs from subfolders of a folder and returns a dictionary of APIs.

    Args:
        path (str): The path to the folder containing the APIs.

    Returns:
        dict: A dictionary containing the API data.
    """
    api_dict = {}
    entries = sorted(os.listdir(path))
    for entry in entries:
        subfolder_path = os.path.join(path, entry)

        # Check if this is a directory
        if os.path.isdir(subfolder_path):
            api_dict[entry] = read_apis_from_folder(subfolder_path)
        logging.info(f"Processed subfolder {entry}")
        logging.info(f"Total # of tools: {len(api_dict)}")
    logging.info(f"Total # of categories: {len(api_dict)}")
    logging.info(
        f"Total # of tools: {sum([len(api_dict[category]) for category in api_dict])}"
    )

    # check total number of apis
    # apis are stored in api_dict[category][tool]["api_list"]
    total_apis = 0
    for category in api_dict:
        for tool in api_dict[category]:
            total_apis += len(api_dict[category][tool]["api_list"])
    logging.info(f"Total # of APIs: {total_apis}")

    return api_dict


def main():
    data_folder = os.path.join(
        os.getenv("TOOLBENCH_DIR"), "toolenv", "tools"
    )
    api_data = read_apis(data_folder)
    store_api_data(api_data, os.getenv("API_DATA_STORE"))


if __name__ == "__main__":
    load_dotenv(".env")
    main()
