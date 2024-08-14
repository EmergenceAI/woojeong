import os
import pickle
import faiss
import pickle
import numpy as np
from typing import List


class ToolRetriever:
    def __init__(
        self,
        dataset: str,
        sim_metric: str = "cosine",
        embedding_mode="openai",
        api_summary_mode="raw",
    ):
        embed_dir = f"/Users/woojeong/Desktop/woojeong/toolbench_analysis/data/api_embeddings_{dataset}"
        try:
            query_embed = pickle.load(
                open(
                    os.path.join(embed_dir, f"id2query_embed_{embedding_mode}.pkl"),
                    "rb",
                )
            )
        except:
            query_embed = None
        api_embed = pickle.load(
            open(
                os.path.join(
                    embed_dir, f"id2api_embed_{api_summary_mode}_{embedding_mode}.pkl"
                ),
                "rb",
            )
        )

        # stack api_embed
        apis = np.stack(list(api_embed.values()), axis=0).astype(np.float32)
        apiidx2id = {i: k for i, k in enumerate(api_embed.keys())}
        d = apis.shape[1]

        if sim_metric == "cosine":
            index = faiss.IndexFlatIP(d)
            faiss.normalize_L2(apis)
            index.add(apis)
        elif sim_metric == "euclidean":
            index = faiss.IndexFlatL2(d)
            index.add(apis)
        else:
            raise ValueError(f"sim_metric {sim_metric} not supported")

        self.d = d
        self.embedding_mode = embedding_mode
        self.sim_metric = sim_metric
        self.query_embed = query_embed
        self.index = index
        self.apiidx2id = apiidx2id

    def call(
        self,
        k: int = 10,
        query_id: int = 0,
        query_text: str = "",
    ) -> List[int]:
        if query_text != "":
            # embed query
            from toolbench_analysis.src.api.embed_apis import embed_texts
            from dotenv import load_dotenv

            load_dotenv(".env")
            query = embed_texts(
                {0: query_text},
                self.embedding_mode,
                embedding_model="text-embedding-3-small",
            )[0]
            query = np.array([query]).astype(np.float32)
            assert (
                query.shape[1] == self.d
            ), f"query shape {query.shape} does not match api shape {self.d}"
        else:
            # get query embed
            assert query_id in self.query_embed, f"query_id {query_id} not found"
            query = np.array([self.query_embed[query_id]]).astype(np.float32)

        if self.sim_metric == "cosine":
            faiss.normalize_L2(query)

        D, I = self.index.search(query, k)
        D, I = D[0], I[0]
        # convert index to id
        I = np.vectorize(lambda x: self.apiidx2id[x])(I)

        return I.tolist()
    
def retrieve_tool(query_text: str, dataset: str = "toolbench") -> List[int]:
    tool_retriever = ToolRetriever(dataset)
    retrieved_apis = tool_retriever.call(query_text=query_text, k=5)
    return retrieved_apis


if __name__ == "__main__":

    tool_retriever = ToolRetriever("toolbench")
    retrieved_apis = tool_retriever.call(query_id=0, k=5)
    retrieved_apis = tool_retriever.call(query_text="This is a dummy query", k=5)
    print(retrieved_apis)
