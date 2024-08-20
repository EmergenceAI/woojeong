# Toolbench Readme
## read and preprocess apis
* read all apis in the toolbench/toolenv/tools directory
* filter the ones with popularityScore over 9.5
```bash
bash scripts/read_and_filter_apis.sh
```
## convert api json -> pandas dataframe and push to hf hub
```bash
python src/api/create_api_hf_dataset.py --push_to_hub
```
* HF dataset - https://huggingface.co/datasets/MerlynMind/toolbench_api

## load queries & anser trajactories
```bash
python src/query/preprocess_answers.py --subset G1 --result_dir data
```
* this will create `data/G1_gt.pkl`

## Embed API docs
```python
python src/api/embed_apis.py --dataset toolbench --summary_mode raw --embedding_mode openai --embedding_model text-embedding-3-small --summary_dir data/api_summaries_toolbench/ --embedding_dir data/api_embeddings_toolbench/
```
* datasets: `toolbench`, `apigen`, `metatool`, `anytoolbench`
* summary_mode: `raw`, `toolbench`, `gpt4-ver1`

# Deprecated
## load queries & answer trajactories -> pandas dataframe and push to hf hub
```bash
python src/query/create_api_hf_dataset.py --subset g1 --push_to_hub
```
* HF dataset - https://huggingface.co/datasets/MerlynMind/toolbench_query

## load G1 query-doc relations and push to hf hub
```bash
python src/api/api_query_relation.py --push_to_hub
```
* These are used to train API retriever in the ToolLLM paper
* HF dataset - https://huggingface.co/datasets/MerlynMind/toolbench_query_api_mapping

## load preprocessed datasets
```python
from dotenv import load_dotenv
from src.utils import load_query_api_mapping, load_api_data, load_query_data

load_dotenv(".env")
query_api_mapping, id2doc, id2query = load_query_api_mapping()
api_data = load_api_data()
query_data = load_query_data("g1")
```

