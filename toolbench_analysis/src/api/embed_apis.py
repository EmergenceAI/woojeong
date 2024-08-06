import json
import os
import pickle
import logging
import argparse
import tiktoken
import openai
from dotenv import load_dotenv
from tqdm import tqdm
from toolbench_analysis.src.api.prompts import API_SUMMARY_PROMPT, API_INTENT_PROMPT
from toolbench_analysis.src.api.utils import get_gpt_response
from agent_system.src.tool_datasets import APIGenDataset, ToolbenchDataset, MetaToolDataset, AnyToolbenchDataset


def _create_summary_prompt(
    api_description, prompt_type="intent"
):
    """Create prompt for LLM to generate API summaries for future retrieval.

    Args:
        api_description (str): API description to summarize

    Returns:
        str: API summary prompt
    """
    if prompt_type == "intent":
        system_prompt = API_INTENT_PROMPT
    elif prompt_type == "summary":
        system_prompt = API_SUMMARY_PROMPT
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": api_description})
    return messages


def create_raw_api_description_apigen(api_info):
    """Create raw API description from data for a given api.
    This raw description will subsequently be sent to the LLM for summarization.

    Args:
        api_info (dict): Dictionary containing the API data
    """
    api_data_subset = {
        k: api_info[k] for k in (
            'name',
            'description',
            'parameters'
        )
    }
    return json.dumps(api_data_subset)

def create_raw_api_description_metatool(api_info):
    """Create raw API description from data for a given api.
    This raw description will subsequently be sent to the LLM for summarization.

    Args:
        api_info (dict): Dictionary containing the API data
    """
    return json.dumps(api_info)


def create_raw_api_description_toolbench(api_info):
    """Create raw API description from data for a given api.
    This raw description will subsequently be sent to the LLM for summarization.

    Args:
        api_info (dict): Dictionary containing the API data
    """
    api_data_subset = {
        k: api_info[k] for k in (
            'category_name',
            'tool_name',
            'tool_description',
            'tool_title',
            'api_name',
            'api_description',
            'api_required_parameters',
            'api_optional_parameters'
        )
    }
    return json.dumps(api_data_subset)


def toolbench_concat_info(api_info):
    """Concatenate toolbench API information into a single string.
    
    Args:
        api_info (dict): Dictionary containing the API data
    """
    doc = (
        str(api_info.get("category_name", "") or "")
        + ", "
        + str(api_info.get("tool_name", "") or "")
        + ", "
        + str(api_info.get("api_name", "") or "")
        + ", "
        + str(api_info.get("api_description", "") or "")
        + ", required_params: "
        + json.dumps(api_info.get("required_parameters", ""))
        + ", optional_params: "
        + json.dumps(api_info.get("optional_parameters", ""))
        + ", return_schema: "
        + json.dumps(api_info.get("template_response", ""))
    )
    return doc


def truncate_texts(texts: list, max_tokens: int=8192, encoding: str="cl100k_base"):
    """Truncate texts to a maximum number of tokens.

    Args:
        texts (list): List of texts to truncate
        max_tokens (int): Maximum number of tokens
        encoding (str): Encoding to use
    """
    encoding = tiktoken.get_encoding(encoding)
    truncated_texts = []
    for text in texts:
        tokens = encoding.encode(text)
        truncated_tokens = tokens[:max_tokens]
        if len(truncated_tokens) < len(tokens):
            print(f"Truncated tokens {len(tokens)} -> {len(truncated_tokens)}")
        truncated_text = encoding.decode(truncated_tokens)
        truncated_texts.append(truncated_text)
    return truncated_texts


def embed_texts(id2text, embedding_mode, embedding_model="text-embedding-3-small"):
    """
    Embed texts using the specified embedding model.
    
    Args:
        id2doc (dict): Dictionary containing the text data
        embedding_mode (str): Embedding mode to use, either "openai" or "toolbench-retriever"
        embedding_model (str): OpenAI model to use for embedding
    """
    if embedding_mode == "openai":
        # truncate texts before embedding    
        id2text = dict(zip(id2text.keys(), truncate_texts(id2text.values())))

        # embed with openai model
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        def batch_embed(id2text, batch_size=128, model=embedding_model):
            embeddings = []
            for i in tqdm(range(0, len(id2text), batch_size)):
                batch_texts = [id2text[id] for id in list(id2text)[i:i+batch_size]]
                batch_embeddings = client.embeddings.create(input = batch_texts, model=model).data
                embeddings.extend(batch_embeddings)
            # convert to dict
            id2embedding = {id: embedding.embedding for id, embedding in zip(id2text.keys(), embeddings)}
            return id2embedding

        # WARNING: This will call the OpenAI API and consume credits
        id2text_embed = batch_embed(id2text)
    elif embedding_mode == "toolbench-retriever":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("ToolBench/ToolBench_IR_bert_based_uncased")
        def batch_embed(id2text, batch_size=128):
            embeddings = []
            for i in tqdm(range(0, len(id2text), batch_size)):
                batch_texts = [id2text[id] for id in list(id2text)[i:i+batch_size]]
                batch_embeddings = model.encode(batch_texts)
                embeddings.extend(batch_embeddings)
            # convert to dict
            id2embedding = {id: embedding for id, embedding in zip(id2text.keys(), embeddings)}
            return id2embedding
        id2text_embed = batch_embed(id2text)
    else:
        raise ValueError(f"Invalid embedding mode: {embedding_mode}")
    return id2text_embed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["toolbench", "apigen", "metatool", "anytoolbench"])
    parser.add_argument("--embed_subset", action="store_true")
    parser.add_argument("--summary_mode", type=str, required=True, choices=["raw", "toolbench", "gpt4-ver1"])
    parser.add_argument("--summary_model", type=str, default="gpt-4-turbo-preview")
    parser.add_argument("--embedding_mode", type=str, default="openai", choices=["openai", "toolbench-retriever"])
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
    parser.add_argument("--summary_dir", type=str, default="data/api_summaries_toolbench/")
    parser.add_argument("--embedding_dir", type=str, default="data/api_embeddings_toolbench/")
    return parser.parse_args()


def main(args):
    if args.dataset == "toolbench":
        logging.info("Using Toolbench dataset")
        ds = ToolbenchDataset()
        summary_mode = "summary"
        create_description_func = create_raw_api_description_toolbench
    elif args.dataset == "apigen":
        logging.info("Using APIGen dataset")
        ds = APIGenDataset()
        summary_mode = "intent"
        create_description_func = create_raw_api_description_apigen
    elif args.dataset == "metatool":
        logging.info("Using MetaTool dataset")
        ds = MetaToolDataset()
        summary_mode = "intent"
        create_description_func = create_raw_api_description_metatool
    elif args.dataset == "anytoolbench":
        logging.info("Using AnyToolbench dataset")
        ds = AnyToolbenchDataset()
        summary_mode = "summary"
        create_description_func = create_raw_api_description_toolbench
    api_data: dict = ds.get_api_data()

    # # filter out the target doc ids
    # if args.embed_subset:
    #     split_info_list = pickle.load(open("data/split_info_list.pkl", "rb"))
    #     target_doc_ids = split_info_list[0]['docs'].tolist()
    #     docid2api = {k: v for k, v in docid2api.items() if k in target_doc_ids}
    #     logging.info(f"Embedding subset of {len(target_doc_ids)} docs")
    
    # === summaries
    os.makedirs(args.summary_dir, exist_ok=True)
    if args.summary_mode == "raw":
        api_summaries = {}
        for id, api_info in api_data.items():
            api_summaries[id] = create_description_func(api_info)
    elif args.summary_mode == "toolbench":
        assert args.dataset == "toolbench", "Toolbench dataset must be used for toolbench summary mode"
        api_summaries = {}
        for id, api_info in api_data.items():
            api_summaries[id] = toolbench_concat_info(api_info)
    elif args.summary_mode == "gpt4-ver1":
        api_summaries_path = os.path.join(args.summary_dir, f"{args.summary_mode}_{len(api_data)}.pkl")

        # create or load api summaries
        if os.path.exists(api_summaries_path):
            logging.info(f"Loading api summaries from {api_summaries_path}")
            with open(api_summaries_path, "rb") as f:
                api_summaries = pickle.load(f)
        else:
            logging.info(f"Generating api summaries using {args.summary_model}")
            # save empty dictionary first
            api_summaries = {}
            with open(api_summaries_path, "wb") as f:
                pickle.dump(api_summaries, f)
        
        # check if any summaries are missing
        if len(api_summaries) == len(api_data):
            logging.info("All api summaries are already generated")
        else:
            logging.info(f"{len(api_data) - len(api_summaries)} api summaries are missing")
            # generate missing summaries
            for i, (api_id, api_info) in enumerate(tqdm(api_data.items())):
                if api_id in api_summaries:
                    continue
                api_description = create_description_func(api_info)
                messages = _create_summary_prompt(api_description, prompt_type=summary_mode)
                response = get_gpt_response(messages, model=args.summary_model)
                api_summaries[api_id] = response

                # save periodically
                if i % 10 == 0:
                    with open(api_summaries_path, "wb") as f:
                        pickle.dump(api_summaries, f)
                    logging.info(f"Saved api summaries to {api_summaries_path}")
    else:
        raise ValueError(f"Invalid summary mode: {args.summary_mode}")
    # breakpoint()

    # === embeddings
    os.makedirs(args.embedding_dir, exist_ok=True)
    
    # embed apis
    save_path = os.path.join(args.embedding_dir, f"id2api_embed_{args.summary_mode}_{args.embedding_mode}_{len(api_data)}.pkl")
    if os.path.exists(save_path):
        logging.info(f"API embeddings already exist at {save_path}")
    else:
        id2api_embed = embed_texts(
            api_summaries,
            embedding_mode=args.embedding_mode,
            embedding_model=args.embedding_model
        )
        with open(save_path, "wb") as f:
            pickle.dump(id2api_embed, f)
        logging.info(f"Saved API embeddings to {save_path}")

    # embed queries
    save_path = os.path.join(args.embedding_dir, f"id2query_embed_{args.embedding_mode}.pkl")
    if os.path.exists(save_path):
        logging.info(f"Query embeddings already exist at {save_path}")
    else:
        id2query_embed = embed_texts(
            ds.get_id2query(),
            embedding_mode=args.embedding_mode,
            embedding_model=args.embedding_model
        )
        with open(save_path, "wb") as f:
            pickle.dump(id2query_embed, f)
        logging.info(f"Saved Query embeddings to {save_path}")


if __name__ == "__main__":
    load_dotenv(".env")
    args = parse_args()
    main(args)