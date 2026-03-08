import os
import argparse
import json
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
import transformers

import src.index
import src.contriever
import src.utils
# import src.slurm   # CPU/Windows: do NOT init distributed
import src.data
from src.evaluation import calculate_matches
import src.normalize_text

from utils import save_file_jsonl, save_file_json

os.environ["TOKENIZERS_PARALLELISM"] = "true"

PROMPT_TEMPLATE = dict(
    RETRIEVAL_PROMPT_TEMPALTE="""Question: {}\nAnswer: {}"""
)

class Retriever:
    def __init__(self, args, model=None, tokenizer=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        # Choose device once; use CPU when CUDA is unavailable (Windows CPU torch)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def embed_queries(self, args, queries):
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):

                # Build query text from various dataset schemas
                if isinstance(q, dict):
                    if "context" in q:
                        query = q["context"]
                        if "question" in q:
                            query = query + "\n" + q["question"]
                    elif "question" in q:
                        query = q["question"]
                    elif "premise" in q:
                        query = q["premise"]
                    else:
                        query = json.dumps(q, ensure_ascii=False)
                else:
                    query = q  # assume string

                if args.lowercase and isinstance(query, str):
                    query = query.lower()
                if args.normalize_text and isinstance(query, str):
                    query = src.normalize_text.normalize(query)

                batch_question.append(query)

                if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:
                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=args.question_maxlength,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {kk: vv.to(self.device) for kk, vv in encoded_batch.items()}
                    output = self.model(**encoded_batch)  # (bs, dim)
                    embeddings.append(output.cpu())
                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")
        return embeddings.numpy()

    def embed_queries_demo(self, queries):
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):
                batch_question.append(q)

                if len(batch_question) == 16 or k == len(queries) - 1:
                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=200,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {kk: vv.to(self.device) for kk, vv in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())
                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")
        return embeddings.numpy()

    def index_encoded_data(self, index, embedding_files, indexing_batch_size):
        allids = []
        allembeddings = np.array([])
        for i, file_path in enumerate(embedding_files):
            print(f"Loading file {file_path}")
            with open(file_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)

            allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
            allids.extend(ids)
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        while allembeddings.shape[0] > 0:
            allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        print("Data indexing completed.")

    def add_embeddings(self, index, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        index.index_data(ids_toadd, embeddings_toadd)
        return embeddings, ids

    def add_passages(self, query, passages, top_passages_and_scores):
        # add passages to original data
        docs = []
        for i, item in enumerate(top_passages_and_scores):
            docs.append(
                {
                    "id": query[i]["id" if "id" in query[0] else "index"],
                    "passages": [passages[int(doc_id)] for k, doc_id in enumerate(item[0])],
                }
            )
        os.makedirs("./retrieved_docs", exist_ok=True)
        save_file_json(
            docs,
            f"./retrieved_docs/{self.args.passages_source}_{self.args.input_file.split('/')[-1].split('.')[0]}_{self.args.n_docs}.json",
        )
        print("Retrieved docs saved.")
        return docs

    def setup_retriever(self):
        print(f"Loading model from: {self.args.retriever_path}")
        self.model, self.tokenizer, _ = src.contriever.load_retriever(self.args.retriever_path)
        self.model.eval()

        # Move to device (CPU on Windows if no CUDA)
        self.model = self.model.to(self.device)

        # half precision only makes sense on CUDA; keep fp32 on CPU
        if (self.device == "cuda") and (not self.args.no_fp16):
            self.model = self.model.half()

        self.index = src.index.Indexer(self.args.projection_size, self.args.n_subquantizers, self.args.n_bits)

        # index all passages
        input_paths = glob.glob(self.args.passages_embeddings)
        input_paths = sorted(input_paths)
        if len(input_paths) == 0:
            raise FileNotFoundError(f"No embedding files matched glob: {self.args.passages_embeddings}")

        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")

        if self.args.save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(self.index, input_paths, self.args.indexing_batch_size)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
            if self.args.save_or_load_index:
                self.index.serialize(embeddings_dir)

        # load passages
        print("loading passages")
        self.passages = src.data.load_passages(self.args.passages)
        if getattr(self.args, "passages_source", None) == "general_knowledge":
            self.passages = [{"id": item["id"], "passages": item["Answer"]} for item in self.passages]
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print("passages have been loaded")

    def search_document(self, query, n_docs=10):
        questions_embedding = self.embed_queries(self.args, query)

        print("Searching documents...")
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(questions_embedding, self.args.n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        return self.add_passages(query, self.passage_id_map, top_ids_and_scores)

    def search_document_demo(self, query, n_docs=10):
        questions_embedding = self.embed_queries_demo([query])
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(questions_embedding, n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
        return self.add_passages(self.passage_id_map, top_ids_and_scores)[:n_docs]

    def setup_retriever_demo(self, retriever_path, passages, passages_embeddings, n_docs=5, save_or_load_index=False):
        print(f"Loading model from: {retriever_path}")
        self.model, self.tokenizer, _ = src.contriever.load_retriever(retriever_path)
        self.model.eval()
        self.model = self.model.to(self.device)

        self.index = src.index.Indexer(768, 0, 8)

        input_paths = glob.glob(passages_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(self.index, input_paths, 1000000)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")

        print("loading passages")
        self.passages = src.data.load_passages(passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print("passages have been loaded")


def add_hasanswer(data, hasanswer):
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r", encoding="utf-8") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data


def main(args):
    retriever = Retriever(args)
    retriever.setup_retriever()
    retrieved_results = retriever.search_document(args.query, args.n_docs)
    save_file_jsonl(retrieved_results, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--query", type=str, default=None, help=".json file containing question and answers")
    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv/.jsonl)")
    parser.add_argument("--passages_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to write retrieved docs")
    parser.add_argument("--n_docs", type=int, default=100, help="Number of documents to retrieve per question")
    parser.add_argument("--validation_workers", type=int, default=32, help="Number of parallel processes")
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument("--save_or_load_index", action="store_true", help="Save/load FAISS index if exists")
    parser.add_argument("--retriever_path", type=str, help="Retriever model path/name")
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Max tokens in a question")
    parser.add_argument("--indexing_batch_size", type=int, default=1000000, help="Batch size for indexing")
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument("--n_subquantizers", type=int, default=0, help="0 => flat index")
    parser.add_argument("--n_bits", type=int, default=8)
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--dataset", type=str, default="none")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")
    parser.add_argument("--passages_source", type=str, default="general_knowledge")
    parser.add_argument("--input_file", type=str, default="input")  # used in add_passages save name

    args = parser.parse_args()

    # IMPORTANT: CPU/Windows => do NOT call src.slurm.init_distributed_mode(args)
    # src.slurm.init_distributed_mode(args)

    main(args)