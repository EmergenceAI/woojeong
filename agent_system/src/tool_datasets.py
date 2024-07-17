import os
import pickle
import numpy as np
from itertools import chain
import json
from toolbench_analysis.src.utils import load_query_api_mapping, load_api_data, load_query_data 
from dataclasses import dataclass

@dataclass
class Dataset():
    id2query: dict = None
    query2apis: dict = None
    api_data: dict = None
    api_data_with_query: dict = None
    query2answers: dict = None
    api_name2id: dict = None

    def get_api_data(self):
        return self.api_data
    
    def get_api_ids(self):
        return list(self.api_data.keys())
    
    def get_api_data_with_query(self):
        return self.api_data_with_query
    
    def get_api_ids_with_query(self):
        return list(self.api_data_with_query.keys())
    
    def get_query2apis(self):
        return self.query2apis
    
    def get_id2query(self):
        return self.id2query
    
    def get_query2answers(self):
        return self.query2answers

    def get_query_by_id(self, qid=0):
        return self.id2query[qid]
    
    def get_api_by_id(self, api_id=0):
        api = self.api_data[api_id]
        return api
    
    def get_api_by_name(self, api_name=""):
        api_id = self.api_name2id[api_name]
        return self.get_api_by_id(api_id)
    
    def get_apis_by_query_id(self, qid=0):
        apis = self.query2apis[qid]
        api_list = [self.get_api_by_id(api_id) for api_id in apis]
        return api_list

    def get_answers_by_query_id(self, qid=0):
        return self.query2answers[qid]
    
    def __len__(self):
        return len(self.id2query)

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
        self.api_data = api_data.to_dict(orient="records")  # TODO: change to dict
        self.api_data_with_query = api_data[api_data["api_id"].isin(docid2api.values())].to_dict(orient="records")
        self.query2answers = None  # TODO

        print("Dataset Stats:")
        print("Number of queries:", len(id2query))
        print("Number of APIs in total:", len(self.api_data))
        print("Number of APIs with query:", len(self.api_data_with_query))
        print("Number of total query-api pairs:", len(query_api_mapping_df))
        print("Avg number of APIs per query:", np.mean([len(v) for v in query2apis.values()]))

    def filter_ds(self, ds):
        # TODO: implement this
        return ds


class APIGenDataset(Dataset):
    def __init__(self):
        from datasets import load_dataset

        # Login using e.g. `huggingface-cli login` to access this dataset
        ds = load_dataset("Salesforce/xlam-function-calling-60k")['train']
        print("# of queries:", len(ds))
        # NOTE: there is a bug in the dataset - filter valid rows
        ds = self.filter_ds(ds)
        print("# of queries after filtering:", len(ds))
        
        # extract data
        qids = ds["id"]
        nested_apis = [json.loads(tool_list) for tool_list in ds['tools']]
        nested_answers = [json.loads(answer_list) for answer_list in ds['answers']]

        # create mapping from query to apis
        id2query = {row["id"]: row["query"] for row in ds}
        
        # pool of apis
        apis = list(chain(*nested_apis))
        unique_apis = list({api["name"]: api for api in apis}.values())  # all names are unique
        api_names = [api["name"] for api in unique_apis]
        api_data: dict = {i: api for i, api in enumerate(unique_apis)}
        api_name2id: dict = {api["name"]: id for id, api in api_data.items()}

        # answers - actual api calls
        nested_answers = [json.loads(answer_list) for answer_list in ds['answers']]
        answers = list(chain(*nested_answers))
        unique_answers = list({answer["name"]: answer for answer in answers}.values())
        api_names_with_query = [answer["name"] for answer in unique_answers]
        assert set(api_names_with_query).issubset(set(api_names)), "Some APIs in the answers are not in the API data."
        # select from the api_data
        api_data_with_query: dict = {api_name2id[api["name"]]: api for api in unique_answers}

        # create mapping from query to apis
        query2answers = {qid: answer_list for qid, answer_list in zip(ds["id"], nested_answers)}
        
        # create query to answer(api) mapping
        query2api_names = {qid: [api["name"] for api in api_list] for qid, api_list in zip(qids, nested_answers)}
        query2apis = {qid: [api_name2id[name] for name in name_list] for qid, name_list in query2api_names.items()}
        assert len(query2apis) == len(id2query)

        self.id2query = id2query
        self.query2apis = query2apis
        self.api_data = api_data
        self.api_data_with_query = api_data_with_query
        self.api_name2id = api_name2id
        self.query2answers = query2answers

        print("Dataset Stats:")
        print("Number of queries:", len(id2query))
        print("Number of APIs in total:", len(self.api_data))
        print("Number of APIs with query:", len(self.api_data_with_query))
        print("Number of total query-api pairs:", np.sum([len(v) for v in query2answers.values()]))
        print("Avg number of APIs per query:", np.mean([len(v) for v in query2answers.values()]))

    def filter_ds(self, ds):
        """
        Filter out rows with invalid function names (not in tools) or empty arguments.
        """
        invalid_ids = set()

        def is_row_valid(example):
            is_valid = True
            
            tool_name_to_optional_flag = {}
            for tool in json.loads(example['tools']):
                has_optional_param = any(
                    'optional' in param['type'] or param.get('default', False) 
                    for param in tool['parameters'].values()
                )
                tool_name_to_optional_flag[tool['name']] = has_optional_param
            
            answers = json.loads(example['answers'])
            if len(answers) == 0:  # No API calls
                is_valid = False
            for ans in answers:
                has_optional_param = tool_name_to_optional_flag.get(ans['name'])
                if (
                    (has_optional_param is None) # Invalid function name! Function doesn't exist.
                    or (not has_optional_param and not ans['arguments']) # Function contains required args but args is empty!
                ):
                    is_valid = False
                    invalid_ids.add(example['id'])
                    break
            return is_valid

        return ds.filter(is_row_valid)



