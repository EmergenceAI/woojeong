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

## load queries & answer trajactories -> pandas dataframe and push to hf hub
```bash
python src/query/create_api_hf_dataset.py --subset g1 --push_to_hub
```