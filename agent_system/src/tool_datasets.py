import os
import pickle
import numpy as np
import pandas as pd
from io import StringIO
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
    
    def get_api_data_df(self):
        return pd.DataFrame(self.api_data).T
    
    def get_api_data_with_query_df(self):
        return pd.DataFrame(self.api_data_with_query).T
    
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
    
    def get_apis_id_by_feature(self, feature_name="", feature_value="", with_query=False):
        if with_query:
            api_pool = self.get_api_data_with_query()
        else:
            api_pool = self.get_api_data()
        api_ids = [api_id for api_id, api in api_pool.items() if api[feature_name] == feature_value]
        return api_ids
    
    def get_api_ids_by_query_id(self, qid=0):
        return self.query2apis[qid]
    
    def get_apis_by_query_id(self, qid=0):
        apis = self.query2apis[qid]
        api_list = [self.get_api_by_id(api_id) for api_id in apis]
        return api_list

    def get_answers_by_query_id(self, qid=0):
        return self.query2answers[qid]
    
    def get_total_apis(self):
        return list(self.get_api_data().keys())
    
    def get_total_queries(self):
        return list(self.get_id2query().keys())
    
    def __len__(self):
        return len(self.id2query)


class ToolbenchDataset(Dataset):
    def __init__(self, subset="G1"):
        # === load data
        toolbench_data_folder = "/Users/woojeong/Desktop/woojeong/toolbench_analysis/data/"
        gt_path = os.path.join(toolbench_data_folder, f"{subset}_gt.pkl")
        assert os.path.exists(gt_path), f"File {gt_path} does not exist, run preprocess_answers.py first"
        with open(gt_path, "rb") as f:
            gt = pickle.load(f)
        # gt keys: ['query', 'final_answer', 'traces', 'functions', 'tool_calls', 'api_ids']

        id2query = {qid: data["query"] for qid, data in gt.items()}
        query2apis = {qid: data["api_ids"] for qid, data in gt.items()}
        query2answers = {qid: data["tool_calls"] for qid, data in gt.items()}
        # for each tool_call, change 'args' to 'arguments'
        for _, tool_calls in query2answers.items():
            for tool_call in tool_calls:
                tool_call["arguments"] = json.loads(tool_call.pop("args"))

        unique_apis = list(set(chain(*query2apis.values())))

        api_data = load_api_data()
        api_data = {i: api for i, api in enumerate(api_data.to_dict(orient="records"))}
        api_data_with_query = {api_id: api_data[api_id] for api_id in unique_apis}

        self.id2query = id2query
        self.query2apis = query2apis
        self.api_data = api_data
        self.api_data_with_query = api_data_with_query
        self.query2answers = query2answers

        print("Dataset Stats:")
        print("Number of queries:", len(id2query))
        print("Number of APIs in total:", len(self.api_data))
        print("Number of APIs with query:", len(self.api_data_with_query))
        print("Number of total query-api pairs:", np.sum([len(v) for v in query2apis.values()]))
        print("Avg number of APIs per query:", np.mean([len(v) for v in query2apis.values()]))


class ToolbenchRetrievalDataset(Dataset):
    def __init__(self, split="concat", load_query_data=False):
        # === load data
        # toolbench_data_folder = "/Users/woojeong/Desktop/woojeong/toolbench_analysis/data/"
        query_api_mapping_df, id2doc, id2query = load_query_api_mapping()
        api_data = load_api_data()

        # === filter query-api mapping that are in the api_data
        docid2api = self.generate_filtered_doc2api_mapping(id2doc, api_data)
        query_api_mapping_df = query_api_mapping_df[query_api_mapping_df["docid"].isin(docid2api.keys())]

        # === convert api index to id
        # api_data should be dictionary with key as index
        api_data = {i: api for i, api in enumerate(api_data.to_dict(orient="records"))}

        # === preprocess
        # reset qid to start from 0
        query_api_mapping_df["qid"] = query_api_mapping_df["qid"] - query_api_mapping_df["qid"].min()
        # create id2query mapping
        id2query = {v["qid"]: v["query"] for v in query_api_mapping_df.to_dict(orient="records")}
        # add api index
        query_api_mapping_df["api"] = query_api_mapping_df["docid"].apply(lambda x: docid2api[x])
        # make these to dict of qid: apis
        query2apis = query_api_mapping_df[["qid", "api"]].groupby("qid")["api"].apply(list).to_dict()

        # === filter by split
        if split == "train":
            query_api_mapping_df = query_api_mapping_df[query_api_mapping_df["split"] == "train"]
        elif split == "test":
            query_api_mapping_df = query_api_mapping_df[query_api_mapping_df["split"] == "test"]
        elif split == "concat":
            pass
        else:
            raise ValueError("Invalid split value. Choose from 'train', 'test', 'concat'")
        filtered_queries = query_api_mapping_df["qid"].values
        id2query = {qid: query for qid, query in id2query.items() if qid in filtered_queries}
        query2apis = {qid: apis for qid, apis in query2apis.items() if qid in filtered_queries}

        # === load query data if needed
        if load_query_data:
            self.query_data = load_query_data("g1")

        # === filter apis with query
        # flatten the list of apis
        unique_apis = list(set(chain(*query2apis.values())))
        api_data_with_query = {api_id: api_data[api_id] for api_id in unique_apis}
        
        self.id2query = id2query
        self.query2apis = query2apis
        self.api_data = api_data
        self.api_data_with_query = api_data_with_query
        self.query2answers = None  # TODO

        print("Dataset Stats:")
        print("Number of queries:", len(id2query))
        print("Number of APIs in total:", len(self.api_data))
        print("Number of APIs with query:", len(self.api_data_with_query))
        print("Number of total query-api pairs:", len(query_api_mapping_df))
        print("Avg number of APIs per query:", np.mean([len(v) for v in query2apis.values()]))

    def generate_filtered_doc2api_mapping(self, id2doc, api_data):
        # apis that are not found in the api_data
        not_found_docids = []

        # map doc_id -> api_id (index in api_data)
        correct_mapping  = {}
        manual_mapping = {
            1105: 34135,
            3076: 6,
            5867: 33854,
            5870: 33871,
        }
        for id, doc in id2doc.items():
            api_info = json.loads(doc)
            category, tool_name, api_name = api_info["category_name"], api_info["tool_name"], api_info["api_name"]
            
            # first check with the names
            data = api_data[(api_data["category_name"] == category) & (api_data["tool_name"] == tool_name) & (api_data["api_name"] == api_name)]
            if len(data) != 0:
                correct_mapping[id] = data.index[0]
                continue

            # if not found, check with the description
            data = api_data[(api_data["api_description"] == api_info["api_description"])]
            if len(data) == 1:
                # if only one match, add to correct_mapping
                correct_mapping[id] = data.index[0]
                continue
            elif len(data) > 1:
                # if multiple matches, check if id is in manual_mapping
                if id in manual_mapping:
                    correct_mapping[id] = manual_mapping[id]
                else:
                    not_found_docids.append(id)
                continue
        
            # neither, add to not_found_apis
            not_found_docids.append(id)
        return correct_mapping


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
        api_data_with_query: dict = {api_name2id[name]: api_data[api_name2id[name]] for name in api_names_with_query}

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


class MetaToolDataset(Dataset):
    def __init__(self, split="concat"):
        # === load dataset
        # clone https://github.com/HowieHwong/MetaTool/tree/master first
        metatool_folder = "/Users/woojeong/Desktop/MetaTool/dataset"
        # single
        single_tool_data = pd.read_csv(os.path.join(metatool_folder, "data/all_clean_data.csv"))
        single_tool_data.rename(columns={"Query": "query", "Tool": "tool"}, inplace=True)
        single_tool_data["split"] = "single"
        # multi
        with open(os.path.join(metatool_folder, "data/multi_tool_query_golden.json"), "rb") as f:
            multi_tool_data = json.load(f)
        multi_tool_data = pd.read_json(StringIO(json.dumps(multi_tool_data)))
        multi_tool_data = multi_tool_data.explode("tool")
        multi_tool_data["split"] = "multi"
        # merge
        tool_data = pd.concat([single_tool_data, multi_tool_data])

        # === filter by split
        if split == "single":
            print("loading queries with single tools")
            tool_data = tool_data[tool_data["split"] == "single"]
        elif split == "multi":
            print("loading queries with multi tools")
            tool_data = tool_data[tool_data["split"] == "multi"]
        elif split == "concat":
            print("loading all queries")
            tool_data = tool_data
        else:
            raise ValueError("Invalid split value. Choose from 'single', 'multi', 'concat'")
        # group by query
        tool_data = tool_data.groupby("query")["tool"].apply(list).reset_index()
        tool_data["qid"] = tool_data.index
        
        # merge tools
        single_tools = set(single_tool_data['tool'].unique())
        multi_tools = set(multi_tool_data['tool'].unique())
        all_tools = single_tools.union(multi_tools)

        # load tool info
        with open(os.path.join(metatool_folder, "plugin_info.json"), "rb") as f:
            tool_info = json.load(f)
        tool_info = pd.read_json(StringIO(json.dumps(tool_info)))
        tool_info.drop("description_for_model", axis=1, inplace=True)
        tool_info.rename(columns={
            "name_for_model": "standard_tool_name",
            "name_for_human": "tool_name",
            "description_for_human": "tool_description",
        }, inplace=True)
        
        # merged tool info (higher level)
        with open(os.path.join(metatool_folder, "big_tool_des.json"), "rb") as f:
            merged_tool_info = json.load(f)
        df_info = []
        for tool_name, tool_description in merged_tool_info.items():
            df_info.append({
                "standard_tool_name": tool_name,
                "tool_description": tool_description
            })
        merged_tool_info = pd.DataFrame(df_info)

        # concat
        tool_info = pd.concat([tool_info, merged_tool_info], ignore_index=True)

        # sanity check
        assert all_tools.issubset(set(tool_info["standard_tool_name"].unique()))

        # === create mappings
        id2query = {row["qid"]: row["query"] for row in tool_data.to_dict(orient="records")}
        api_data = {i: api for i, api in enumerate(tool_info.to_dict(orient="records"))}
        api_name2id = {api["standard_tool_name"]: id for id, api in api_data.items()}
        query2apis = {qid: [api_name2id[api] for api in apis] for qid, apis in tool_data[["qid", "tool"]].values}
        assert len(query2apis) == len(id2query)
        api_data_with_query = {api_id: api_data[api_id] for api_id in set(chain(*query2apis.values()))}

        self.id2query = id2query
        self.query2apis = query2apis
        self.api_data = api_data
        self.api_data_with_query = api_data_with_query
        self.api_name2id = api_name2id
        self.query2answers = None

        print("Dataset Stats:")
        print("Number of queries:", len(id2query))
        print("Number of APIs in total:", len(self.api_data))
        print("Number of APIs with query:", len(self.api_data_with_query))
        print("Number of total query-api pairs:", np.sum([len(v) for v in query2apis.values()]))
        print("Avg number of APIs per query:", np.mean([len(v) for v in query2apis.values()]))


class AnyToolbenchDataset(Dataset):
    def __init__(self):
        # === load data
        # clone https://github.com/dyabel/AnyTool/tree/public first
        path = "/Users/woojeong/Desktop/AnyTool/atb_data/anytoolbench.json"
        with open(path, "rb") as f:
            data = json.load(f)
        api_data = load_api_data()  # toolbench api data
        
        # fix minor issues
        # 'Get Articles by Date' -> 'Get Articles by  Date'
        data[58]['gt_api_list'][1]['api_name'] = 'Get Articles by  Date'
        # 'Text Sentiment Analysis' -> 'Text Sentiment Analysis '
        data[92]['gt_api_list'][1]['tool_name'] = 'Text Sentiment Analysis '
        
        # query mapping
        id2query = {int(row["query_id"]): row["query"] for row in data}
        
        # query to api mapping
        query2apis = {}
        for i, row in enumerate(data):
            query_id = int(row['query_id'])
            api_list = row['gt_api_list']
            api_indices = []
            for api in api_list:
                match = api_data[
                    (api_data['category_name'] == api['category_name']) 
                    & (api_data['tool_name'] == api['tool_name']) 
                    & (api_data['api_name'] == api['api_name'])]
                assert len(match) == 1
                api_indices.append(match.index[0])
            query2apis[query_id] = api_indices

        # === convert api index to id
        # api_data should be dictionary with key as index
        api_data = {i: api for i, api in enumerate(api_data.to_dict(orient="records"))}
        # merge "Finance" into "Financial"
        for api in api_data.values():
            if api["category_name"] == "Finance":
                api["category_name"] = "Financial"

        # === filter apis with query
        # flatten the list of apis
        unique_apis = list(set(chain(*query2apis.values())))
        api_data_with_query = {api_id: api_data[api_id] for api_id in unique_apis}
        
        self.id2query = id2query
        self.query2apis = query2apis
        self.api_data = api_data
        self.api_data_with_query = api_data_with_query
        self.query2answers = None  # TODO

        print("Dataset Stats:")
        print("Number of queries:", len(id2query))
        print("Number of APIs in total:", len(self.api_data))
        print("Number of APIs with query:", len(self.api_data_with_query))
        print("Number of total query-api pairs:", len(list(chain(*query2apis.values()))))
        print("Avg number of APIs per query:", np.mean([len(v) for v in query2apis.values()]))
