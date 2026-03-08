
# Replicating Causal Reasoning in LLMs under Resource-Constrained Settings

This repository reproduces and extends experiments from  
**"Unveiling Causal Reasoning in Large Language Models: Reality or Mirage?" (NeurIPS 2024)**.

Our project evaluates causal reasoning performance of large language models on the **CausalProbe-2024 benchmark**, focusing on four inference strategies:

- Vanilla prompting
- Chain-of-Thought (CoT)
- Retrieval-Augmented Generation (RAG)
- G²-Reasoner

Unlike the original implementation, our experiments are conducted in a **CPU-compatible local environment** using **Ollama API access to DeepSeek-v3.1**, enabling more accessible and reproducible evaluation without requiring CUDA GPUs.

---

# Project Structure

CausalProbe-2024/
│
├── benchmarks/
│   └── CausalProbe2024/
│       ├── CausalProbe-E.json
│       └── CausalProbe-H.json
│
├── kb/
│   ├── general_knowledge_passages.jsonl
│   └── general_knowledge_embeddings/
│
├── result_logs/
│   ├── E/
│   └── H/
│
├── retrieved_docs/
│
├── main_ollama.py
├── passage_retrieval.py
├── generate_embeddings.py
├── prompts.py
├── utils.py
│
└── rerun_no_choice_by_id.py

---

# Environment Setup

Install required dependencies:

pip install -r requirements.txt

---

# Retrieval Knowledge Base

We use the **General Knowledge QA dataset** as the retrieval corpus.

Each entry contains:

{
 "id": 1,
 "Question": "...",
 "Answer": "..."
}

Only the **Answer field** is used as retrieval knowledge.

---

# Building the Retrieval Index

python generate_embeddings.py \
  --model_name_or_path facebook/contriever-msmarco \
  --output_dir kb/general_knowledge_embeddings \
  --prefix passages \
  --passages kb/general_knowledge_passages.jsonl \
  --shard_id 0 \
  --num_shards 1 \
  --per_gpu_batch_size 16 \
  --passage_maxlength 256

---

# Running Experiments

## Vanilla

python main_ollama.py \
  --model_name deepseek-v3.1:671b-cloud \
  --input_file benchmarks/CausalProbe2024/CausalProbe-E.json \
  --mode vanilla \
  --batch_size 1 \
  --max_new_tokens 50 \
  --metric multiple_choice_match \
  --prompt_name prompt_mcqa_causalprobe \
  --task qa \
  --result_fp_base result_logs/E/

---

## Chain-of-Thought

python main_ollama.py \
  --model_name deepseek-v3.1:671b-cloud \
  --input_file benchmarks/CausalProbe2024/CausalProbe-E.json \
  --mode vanilla \
  --batch_size 1 \
  --max_new_tokens 128 \
  --metric multiple_choice_match \
  --prompt_name prompt_mcqa_cot_causalprobe \
  --task qa \
  --result_fp_base result_logs/E/

---

## RAG

python main_ollama.py \
  --model_name deepseek-v3.1:671b-cloud \
  --input_file benchmarks/CausalProbe2024/CausalProbe-E.json \
  --mode retrieval \
  --batch_size 1 \
  --max_new_tokens 50 \
  --metric multiple_choice_match \
  --prompt_name prompt_mcqa_retrieval_causalprobe \
  --task qa \
  --result_fp_base result_logs/E/ \
  --passages kb/general_knowledge_passages.jsonl \
  --passages_embeddings kb/general_knowledge_embeddings/passages_* \
  --passages_source general_knowledge \
  --retriever_path facebook/contriever-msmarco

---

## G²-Reasoner

python main_ollama.py \
  --model_name deepseek-v3.1:671b-cloud \
  --input_file benchmarks/CausalProbe2024/CausalProbe-E.json \
  --mode retrieval \
  --batch_size 1 \
  --max_new_tokens 50 \
  --metric multiple_choice_match \
  --prompt_name prompt_mcqa_g2reasoner_causalprobe \
  --task qa \
  --result_fp_base result_logs/E/ \
  --passages kb/general_knowledge_passages.jsonl \
  --passages_embeddings kb/general_knowledge_embeddings/passages_* \
  --passages_source general_knowledge \
  --retriever_path facebook/contriever-msmarco

---

# Error Handling

Some outputs may contain:

- No choice id is outputted
- HTTP 429 errors
- HTTP 500 errors

We provide a rerun pipeline:

rerun_no_choice_by_id.py

This script:

1. Detects problematic samples
2. Creates a subset benchmark
3. Reruns them
4. Merges the corrected outputs

---

# Benchmark

We evaluate:

| Dataset | Description |
|-------|------|
| CausalProbe-E | Easier causal reasoning |
| CausalProbe-H | Hard causal reasoning |

---

# Metric

multiple_choice_match

The predicted choice must match the ground truth answer.

---

# Citation

Chi et al., 2024  
Unveiling Causal Reasoning in Large Language Models: Reality or Mirage?  
NeurIPS 2024
