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

## Todo
- [x] preprocess query-api relations
- [ ] split train/val apis
- [ ] split train/val queries
- [ ] queries in instructions > answers, where are missing answers?