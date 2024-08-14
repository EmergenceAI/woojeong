# Build a large agentic system using AutoGen

## Execution
```bash
python src/main.py --dataset apigen --tool_top_k 20
```
* supports only apigen

## Evaluation
```bash
python src/evaluate.py --dataset apigen --eval_retrieval --eval_tool --eval_solved
```