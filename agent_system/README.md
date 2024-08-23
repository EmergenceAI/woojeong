# Building a Large Agentic System using AutoGen

This directory provides scripts and guidance for constructing and evaluating a large agentic system using AutoGen.

## Benchmarks

### Download

* **Toolbench**
  * Clone the repository: `https://github.com/OpenBMB/ToolBench/tree/master`
  * Download the data into the `data/` folder.
  * Set the `TOOLBENCH_DIR` environment variable (e.g., `"/Users/woojeong/Desktop/ToolBench/data"`).
  * Preprocess the dataset first (refer to `toolbench_analysis` README)

* **APIGen**
  * APIGen data will be automatically downloaded from [Hugging Face](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k).

* **MetaTool**
  * Clone the repository: `https://github.com/HowieHwong/MetaTool/tree/master`
  * Set the `METATOOL_DIR` environment variable (e.g., `"/Users/woojeong/Desktop/MetaTool/dataset"`).

* **AnyTool**
  * Clone the repository: `https://github.com/dyabel/AnyTool/tree/public`
  * Set the `ANYTOOLBENCH_DIR` environment variable (e.g., `"/Users/woojeong/Desktop/AnyTool/atb_data"`).

## Dataset Classes

All dataset classes inherit from a base `Dataset` class, allowing them to be called in a standardized manner. These classes can be found in `agent_system.src.tool_datasets`, and each dataset can be loaded with different configurations.

* **ToolbenchDataset**
  * Used for ToolLLM training.
  * **Options:**
    * `subset`: Specifies different subsets in Toolbench. Preprocess the dataset first (refer to `toolbench_analysis` README). Options: `["G1", "G2", "G3"]`, Default: `"G1"`.
    * `multi_step`: Filters multi-step/multi-tool queries. Default: `False`.

* **ToolbenchRetrievalDataset**
  * Used for training/testing the tool-retriever as described in the Toolbench paper. Contains only queries and ground truth APIs, without tool call information.
  * **Options:**
    * `split`: `"train"`, `"test"`, or `"concat"`.

* **APIGenDataset**
  * **Options:**
    * `multi_step`: Filters multi-step/multi-tool queries. Default: `False`.

* **MetaToolDataset**
  * **Options:**
    * `split`: Follows the split from the original paper. Single or multi-tool options: `["single", "multi", "concat"]`.

* **AnyToolbenchDataset**
  * **Options:**
    * `multi_step`: Filters multi-step/multi-tool queries. Default: `False`.

## Embed API Documents

Embed API documents into fixed-dimension vectors before running the AutoGen system:

```bash
python src/retrieval/embed_apis.py --dataset toolbench --summary_mode raw --embedding_mode openai --embedding_model text-embedding-3-small --summary_dir retrieval_data/api_summaries_toolbench/ --embedding_dir retrieval_data/api_embeddings_toolbench/
```

* **Datasets:** `toolbench`, `apigen`, `metatool`, `anytoolbench`.
* **Summary Mode:**
  * `raw`: Uses category name, tool name, tool description, tool title, API name, API description, required/optional parameters.
  * `toolbench`: Default in Toolbench paper to describe an API: Uses category name, tool name, API name, API description, required/optional parameters, and template response (if available).
  * `gpt4-ver`: Uses GPT-4 to generate a natural language summary of the API's capabilities. Specify `summary_model`.
* **Embedding Mode:**
  * `toolbench-retriever`: Requires running the [Toolbench retriever](https://github.com/OpenBMB/ToolBench?tab=readme-ov-file#model) (a small sentence transformer model) locally. Suitable for GPU machines.
  * `openai`: Uses OpenAI embedding model to embed API documents. Specify `embedding_model` (default: `text-embedding-3-small`).

## Run the Agent System

Load a specified dataset, retrieve the top 20 relevant tools, and run the AutoGen system with a maximum of 20 rounds. Results will be saved as JSON, e.g., `../results/apigen_api20.json`.

```bash
python src/main.py --dataset toolbench --multi_step --tool_top_k 20 --autogen_max_chat_round 20 --result_dir results
```

* This script is currently verified with `apigen` and `toolbench`, but further verification is needed for other datasets.

## Evaluation

Evaluate the results of the agent system. Ensure that the agent system has been run and that the results are saved in the `results` directory for correct loading.

```bash
python src/evaluate.py --dataset apigen --tool_top_k 20 --eval_retrieval --eval_tool --eval_solved
```

* **Warning:** Since multithreading is used, error messages may not be displayed. Set `n_threads=1` for debugging.
* Use the `overwrite_all` flag to overwrite all results. It should be turned off to fill in missing values.
* This code only works for results generated from running `src/main.py`. To evaluate bootstrapping results, manually set `result_file_path` and `eval_result_file_path` inside the code.

## Miscellaneous

Jupyter notebooks used for displaying results and plots can be found in the `notebook` directory.