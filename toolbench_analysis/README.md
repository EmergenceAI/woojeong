# Toolbench Analysis

This directory contains scripts and instructions for preprocessing and analyzing the Toolbench dataset.

## Setup
To ensure smooth execution, make sure to load environment variables correctly. If the code fails to do so, place a `.env` file in the working directory.

## API Data Preprocessing

### 1. Read and Filter APIs
Read all APIs from the `{TOOLBENCH_DIR}/toolbench/toolenv/tools` directory and filter those with a `popularityScore` above 9.5.
Run the following script:

```bash
bash scripts/read_and_filter_apis.sh
```

### 2. Convert API JSON to Pandas DataFrame and Push to Hugging Face Hub
Convert the filtered APIs to a pandas DataFrame and push the dataset to the Huggingface hub:

```bash
python src/api/create_api_hf_dataset.py --push_to_hub
```

* Huggingface Dataset: [toolbench_api](https://huggingface.co/datasets/MerlynMind/toolbench_api)

## Query and Answer Trajectories

### 1. Load and Preprocess Queries & Answer Trajectories
Load queries and answer trajectories for a specific subset (e.g., G1) and save the results to the `data` directory:

```bash
python src/query/preprocess_answers.py --subset G1 --result_dir data
```

* This will create the file `data/G1_gt.pkl` which be used for initializing `ToolbenchDataset` class.

## Deprecated

### 1. Load Queries & Answer Trajectories and Push to Hugging Face Hub
**Note:** This dataset, known as the "Toolbench Retrieval" dataset, was used to train and test the Toolbench retriever. It includes only tool selection ground truth. This process is now deprecated in favor of the above datasets.

```bash
python src/query/create_api_hf_dataset.py --subset g1 --push_to_hub
```

* Huggingface Dataset: [toolbench_query](https://huggingface.co/datasets/MerlynMind/toolbench_query)

### 2. Load G1 Query-Document Relations and Push to Hugging Face Hub
**Note:** We now use a standardized data class for these relations. These mappings were used to train the API retriever as described in the ToolLLM paper.

```bash
python src/api/api_query_relation.py --push_to_hub
```

* Huggingface Dataset: [toolbench_query_api_mapping](https://huggingface.co/datasets/MerlynMind/toolbench_query_api_mapping)