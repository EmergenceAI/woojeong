import os
import pickle
import numpy as np
from itertools import chain
import json
from toolbench_analysis.src.utils import load_query_api_mapping, load_api_data, load_query_data 
from dataclasses import dataclass

@dataclass
class Dataset():
    id2query = None
    query2apis = None
    api_data = None
    api_data_with_query = None

    def get_api_data(self):
        return self.api_data
    
    def get_api_data_with_query(self):
        return self.api_data_with_query
    
    def get_query2apis(self):
        return self.query2apis
    
    def get_id2query(self):
        return self.id2query

    def get_query_by_id(self, qid=0):
        return self.id2query[qid]
    
    def get_api_by_id(self, api_id=0):
        api = self.api_data[api_id]
        assert api["api_id"] == api_id
        return api
    
    def get_apis_by_query_id(self, qid=0):
        apis = self.query2apis[qid]
        api_list = [self.get_api_by_id(api_id) for api_id in apis]
        return api_list


class ToolbenchDataset(Dataset):
    def __init__(self, filter_query_api_mapping=False, load_query_data=False):
        # === load query_api mapping data
        toolbench_data_folder = "/Users/woojeong/Desktop/woojeong/toolbench_analysis/data/"
        if filter_query_api_mapping:
            query_api_mapping_df, id2doc, id2query = load_query_api_mapping(local_file_path=os.path.join(toolbench_data_folder, "filtered_query_api_mapping.csv"))
        else:
            query_api_mapping_df, id2doc, id2query = load_query_api_mapping()
        # TODO: add code to generate docid2api
        with open(os.path.join(toolbench_data_folder, "docid2api.pkl"), "rb") as f:
            docid2api = pickle.load(f)

        # === load api data
        api_data = load_api_data()
        api_data.reset_index(inplace=True)
        api_data = api_data.rename(columns={"index": "api_id"})

        # === preprocess
        # reset qid to start from 0
        query_api_mapping_df["qid"] = query_api_mapping_df["qid"] - query_api_mapping_df["qid"].min()
        # create id2query mapping
        id2query = {v["qid"]: v["query"] for v in query_api_mapping_df.to_dict(orient="records")}
        # add api index
        query_api_mapping_df["api"] = query_api_mapping_df["docid"].apply(lambda x: docid2api[x])
        # make these to dict of qid: apis
        query2apis = query_api_mapping_df[["qid", "api"]].groupby("qid")["api"].apply(list).to_dict()

        # === load query data if needed
        if load_query_data:
            self.query_data = load_query_data("g1")
        
        self.id2query = id2query
        self.query2apis = query2apis
        self.api_data = api_data.to_dict(orient="records")
        self.api_data_with_query = api_data[api_data["api_id"].isin(docid2api.values())]

        print("Dataset Stats:")
        print("Number of queries:", len(id2query))
        print("Number of APIs in total:", len(self.api_data))
        print("Number of APIs with query:", len(self.api_data_with_query))
        print("Number of total query-api pairs:", len(query_api_mapping_df))
        print("Avg number of APIs per query:", np.mean([len(v) for v in query2apis.values()]))



class APIGenDataset(Dataset):
    def __init__(self):
        from datasets import load_dataset

        # Login using e.g. `huggingface-cli login` to access this dataset
        ds = load_dataset("Salesforce/xlam-function-calling-60k")['train']
        
        # create mapping from query to apis
        id2query = {row["id"]: row["query"] for row in ds}
        
        # pool of tools
        nested_tools = [json.loads(tool_list) for tool_list in ds['tools']]
        apis = list(chain(*nested_tools))
        unique_apis = list({api["name"]: api for api in apis}.values())
        # add api_id to each api dict
        for i, api in enumerate(unique_apis):
            api["api_id"] = i
        api_data = unique_apis

        # api names are unique
        api_names = [api["name"] for api in api_data]
        assert len(set(api_names)) == len(api_names)
        api_name_to_id = {api["name"]: api["api_id"] for api in api_data}

        # assign api_id to each api in the nested_tools
        for tool_list in nested_tools:
            for api in tool_list:
                api["api_id"] = api_name_to_id[api["name"]]

        # create query2apis mapping
        qids = ds["id"]
        query2apis = {qid: [api["api_id"] for api in tool_list] for qid, tool_list in zip(qids, nested_tools)}
        assert len(query2apis) == len(id2query)

        # check if the mapping is correct
        query2apis_list = list(query2apis.values())
        # merge nested list
        all_apis = list(chain(*query2apis_list))
        assert len(set(all_apis)) == len(api_data)

        self.id2query = id2query
        self.query2apis = query2apis
        self.api_data = api_data
        self.api_data_with_query = api_data

        print("Dataset Stats:")
        print("Number of queries:", len(id2query))
        print("Number of APIs in total:", len(self.api_data))
        print("Number of APIs with query:", len(self.api_data_with_query))
        print("Number of total query-api pairs:", len(all_apis))
        print("Avg number of APIs per query:", np.mean([len(v) for v in query2apis.values()]))


