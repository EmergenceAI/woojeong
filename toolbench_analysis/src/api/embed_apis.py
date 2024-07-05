import json
import os
import pickle
import logging
import argparse
import tiktoken
import openai
from dotenv import load_dotenv
from tqdm import tqdm
from toolbench_analysis.src.utils import load_query_api_mapping, load_api_data
from toolbench_analysis.src.api.prompts import API_SUMMARY_PROMPT
from toolbench_analysis.src.api.utils import get_gpt_response


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


def create_raw_api_description(api_info):
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

def create_raw_api_description_toolbench(api_info):
    """Create raw API description from data for a given api, according to the toolbench paper
    
    Args:
        api_info (dict): Dictionary containing the API data
    """
    doc = (
        (api_info.get("category_name", "") or "")
        + ", "
        + (api_info.get("tool_name", "") or "")
        + ", "
        + (api_info.get("api_name", "") or "")
        + ", "
        + (api_info.get("api_description", "") or "")
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


def embed_api_summaries(id2doc, embedding_model="text-embedding-3-small"):
    """
    Embed API summaries using OpenAI model.
    
    Args:
        id2doc (dict): Dictionary containing API summaries
        embedding_model (str): OpenAI model to use for embedding
    """
    # truncate texts before embedding    
    id2doc = dict(zip(id2doc.keys(), truncate_texts(id2doc.values())))

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
    id2doc_embed = batch_embed(id2doc)
    return id2doc_embed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_subset", action="store_true")
    parser.add_argument("--summary_mode", type=str, required=True, choices=["raw", "toolbench", "gpt4-ver1"])
    parser.add_argument("--summary_model", type=str, default="gpt-4-turbo-preview")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
    return parser.parse_args()


def main(args):
    # load api info
    api_data = load_api_data()
    # load docid to api mapping
    docid2api = pickle.load(open("data/docid2api.pkl", "rb"))

    # filter out the target doc ids
    if args.embed_subset:
        split_info_list = pickle.load(open("data/split_info_list.pkl", "rb"))
        target_doc_ids = split_info_list[0]['docs'].tolist()
        docid2api = {k: v for k, v in docid2api.items() if k in target_doc_ids}
        logging.info(f"Embedding subset of {len(target_doc_ids)} docs")
    
    # create api summaries
    if args.summary_mode == "raw":
        api_summaries = {}
        for doc_id, api_id in tqdm(docid2api.items()):
            api_info = api_data.iloc[api_id].to_dict()
            api_description = create_raw_api_description(api_info)
            api_summaries[doc_id] = api_description
    elif args.summary_mode == "toolbench":
        api_summaries = {}
        for doc_id, api_id in tqdm(docid2api.items()):
            api_info = api_data.iloc[api_id].to_dict()
            api_description = create_raw_api_description_toolbench(api_info)
            api_summaries[doc_id] = api_description
    elif args.summary_mode == "gpt4-ver1":
        api_summaries_path = f"data/api_summaries/{args.summary_mode}_{len(docid2api)}.pkl"

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
        if len(api_summaries) == len(docid2api):
            logging.info("All api summaries are already generated")
        else:
            logging.info(f"{len(docid2api) - len(api_summaries)} api summaries are missing")
            # generate missing summaries
            for i, (doc_id, api_id) in enumerate(tqdm(docid2api.items())):
                if doc_id in api_summaries:
                    continue
                api_info = api_data.iloc[api_id].to_dict()
                api_description = create_raw_api_description(api_info)
                messages = _create_summary_prompt(api_description)
                response = get_gpt_response(messages, model=args.summary_model)
                api_summaries[doc_id] = response

                # save periodically
                if i % 10 == 0:
                    with open(api_summaries_path, "wb") as f:
                        pickle.dump(api_summaries, f)
                    logging.info(f"Saved api summaries to {api_summaries_path}")
    else:
        raise ValueError(f"Invalid summary mode: {args.summary_mode}")
    # breakpoint()

    # embed api summaries
    id2doc_embed = embed_api_summaries(api_summaries, embedding_model=args.embedding_model)

    # save embeddings
    oai_embedding_dir = "data/api_embeddings/"
    os.makedirs(oai_embedding_dir, exist_ok=True)
    save_path = os.path.join(oai_embedding_dir, f"id2doc_embed_{args.summary_mode}_{len(docid2api)}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(id2doc_embed, f)
    logging.info(f"Saved api summary embeddings to {save_path}")


if __name__ == "__main__":
    load_dotenv(".env")
    args = parse_args()
    main(args)