# Revisiting Causal Reasoning in Large Language Models under Resource-Constrained Inference

This repository reproduces and extends the experiments from:

Chi et al., 2024  
Unveiling Causal Reasoning in Large Language Models: Reality or Mirage?  
NeurIPS 2024

We evaluate causal reasoning performance of large language models on the CausalProbe-2024 benchmark under a resource‑constrained inference setting using API-based LLM access instead of GPU-based local inference.

We compare four inference strategies:

- Vanilla prompting
- Chain-of-Thought (CoT)
- Retrieval-Augmented Generation (RAG)
- G²-Reasoner

Unlike the original implementation, our setup:

- runs without CUDA GPU
- uses Ollama API with DeepSeek‑v3.1
- uses CPU-based retrieval
- includes rerun pipeline for API errors
- evaluates on CausalProbe‑E and CausalProbe‑H


# Overview

| Component | Setting |
|----------|----------|
| LLM | deepseek-v3.1 (Ollama cloud API) |
| Retrieval model | facebook/contriever-msmarco |
| Index | FAISS (CPU) |
| Knowledge base | General Knowledge QA |
| Benchmark | CausalProbe‑2024 |
| Inference | CPU + API |
| Error handling | rerun pipeline |


# Project Structure

```
CausalProbe-2024/
│
├── benchmarks/
│   └── CausalProbe2024/
│       ├── CausalProbe-E.json
│       └── CausalProbe-H.json
│
├── main_ollama.py
├── passage_retrieval.py
├── generate_embeddings.py
├── prompts.py
├── utils.py
├── rerun_no_choice_by_id.py
│
├── README.md
└── requirements.txt
```

Large folders not included in repo:

```
kb/
result_logs/
retrieved_docs/
```


# Installation

pip install -r requirements.txt


# Benchmark

| Dataset | Description |
|----------|------------|
| CausalProbe-E | easier causal reasoning |
| CausalProbe-H | harder causal reasoning |

Metric:

multiple_choice_match


# Build Retrieval Embeddings

python generate_embeddings.py
--model_name_or_path facebook/contriever-msmarco
--output_dir kb/general_knowledge_embeddings
--prefix passages
--passages kb/general_knowledge_passages.jsonl
--shard_id 0
--num_shards 1
--per_gpu_batch_size 16
--passage_maxlength 256


# Vanilla

python main_ollama.py
--model_name deepseek-v3.1:671b-cloud
--input_file [benchmark dir]
--mode vanilla  
--batch_size 1  
--max_new_tokens 128  
--metric multiple_choice_match 
--prompt_name prompt_mcqa_causalprobe  
--task qa  
--result_fp_base result_logs


# CoT

python main_ollama.py 
--model_name deepseek-v3.1:671b-cloud  
--input_file [benchmark dir]
--mode vanilla 
--batch_size 1  
--max_new_tokens 128 
--metric multiple_choice_match  
--prompt_name prompt_mcqa_cot_causalprobe 
--task qa   
--result_fp_base result_logs


# RAG

python main_ollama.py 
--model_name deepseek-v3.1:671b-cloud 
--input_file [benchmark dir]
--mode retrieval 
--batch_size 1 
--max_new_tokens 128 
--metric multiple_choice_match 
--prompt_name prompt_mcqa_retrieval_causalprobe  
--task qa  
--result_fp_base result_logs
--passages kb/general_knowledge_passages.jsonl 
--passages_embeddings kb/general_knowledge_embeddings/passages_* 
--passages_source general_knowledge  
--retriever_path facebook/contriever-msmarco



# G²‑Reasoner

python main_ollama.py 
--model_name deepseek-v3.1:671b-cloud 
--input_file [benchmark dir]
--mode retrieval  
--batch_size 1  
--max_new_tokens 128 
--metric multiple_choice_match
--prompt_name prompt_mcqa_g2reasoner_causalprobe
--task qa 
--result_fp_base result_logs 
--passages kb/general_knowledge_passages.jsonl
--passages_embeddings kb/general_knowledge_embeddings/passages_* 
--passages_source general_knowledge
--retriever_path facebook/contriever-msmarco


# Error Handling

rerun_no_choice_by_id.py

Used to rerun:

- No choice id
- 429 errors
- 500 errors

Pipeline:

1. detect failed samples
2. create subset
3. rerun
4. replace by id
5. save merged file


# Citation

Chi et al., 2024  
Unveiling Causal Reasoning in Large Language Models: Reality or Mirage?  
NeurIPS 2024
