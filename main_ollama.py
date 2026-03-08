# main_ollama.py
# Usage: same as main.py, but model is ALWAYS Ollama.
# Example:
#   python main_ollama.py --model_name deepseek-v3.1:671b-cloud --input_file benchmarks\CausalProbe2024\CausalProbe_E.json ...

import os
import argparse
import numpy as np
from tqdm import tqdm
import ast
import re
import requests

from utils import (
    load_file,
    TASK_INST,
    save_file_jsonl,
    postprocess_answers_closed,
)
from prompts import PROMPT_DICT
from metrics import (
    metric_max_over_ground_truths,
    exact_match_score,
    match,
    binary_choice_match,
    multiple_choice_match,
)
from passage_retrieval import Retriever


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


def call_model_ollama(prompt: str, model: str, max_tokens: int = 50, temperature: float = 0.0) -> str:
    """
    model examples:
      - deepseek-v3.1:671b-cloud
      - qwen2.5:7b
    """
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        return f"ERROR: OLLAMA error outputs {repr(e)}"


def postprocess_output(pred: str, force_choice_letter: bool = False) -> str:
    """
    Keep close to original postprocess_output, but optionally enforce A-D extraction for MCQ metrics.
    """
    if pred is None:
        return ""
    pred = pred.replace("</s>", "")
    pred = pred.replace("\n", "")
    pred = pred.strip()

    if force_choice_letter:
        m = re.search(r"\b([A-D])\b", pred)
        if m:
            return m.group(1)

    return pred


def main():
    parser = argparse.ArgumentParser()

    # IMPORTANT: in this file, model_name is an Ollama model name (NO "ollama:" prefix needed)
    parser.add_argument("--model_name", type=str, required=True,
                        help="Ollama model name, e.g. deepseek-v3.1:671b-cloud or qwen2.5:7b")

    # Retrieval args (same as original)
    parser.add_argument("--retriever_path", type=str, default=None)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--passages", type=str, default=None)
    parser.add_argument("--passages_embeddings", type=str, default=None)
    parser.add_argument("--passages_source", type=str, default=None)
    parser.add_argument("--mode", type=str, default="vanilla")  # vanilla or retrieval

    # generation
    parser.add_argument("--max_new_tokens", type=int, default=50)

    # retriever settings (kept for compatibility)
    parser.add_argument("--int8bit", action="store_true")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument("--n_subquantizers", type=int, default=0)
    parser.add_argument("--n_bits", type=int, default=8)
    parser.add_argument("--indexing_batch_size", type=int, default=1000000)
    parser.add_argument("--save_or_load_index", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--normalize_text", action="store_true")
    parser.add_argument("--question_maxlength", type=int, default=512)
    parser.add_argument("--per_gpu_batch_size", type=int, default=64)
    parser.add_argument("--n_docs", type=int, default=20)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--load_retrieved_docs", type=eval, default=True)

    # evaluation & prompting
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--result_fp_base", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--prompt_name", type=str, default="prompt_no_input")
    parser.add_argument("--batch_size", type=int, default=1)  # for Ollama: keep 1 for stability
    parser.add_argument("--choices", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)

    # optional: force output to single A-D for MCQ matching
    parser.add_argument("--force_choice_letter", action="store_true",
                        help="If enabled, postprocess will extract first A-D from model output (recommended for multiple_choice_match).")

    args = parser.parse_args()

    os.makedirs(args.result_fp_base, exist_ok=True)

    result_fp = (
        args.result_fp_base
        + f"result_{args.model_name.replace(':','_')}_{args.mode}_"
        + f"{os.path.basename(args.input_file).split('.')[0]}_{args.prompt_name[7:]}.json"
    )
    print(f"Results saved in {result_fp}")

    if args.prompt_name not in PROMPT_DICT:
        raise KeyError(
            f"prompt_name='{args.prompt_name}' not found in PROMPT_DICT. "
            f"Run: python -c \"from prompts import PROMPT_DICT; print('\\n'.join(PROMPT_DICT.keys()))\""
        )

    input_data = load_file(args.input_file)

    # Retrieval mode: keep same behavior as repo
    if args.mode == "retrieval":
        if args.load_retrieved_docs:
            retrieved_fp = f"./retrieved_docs/{args.passages_source}_{os.path.basename(args.input_file).split('.')[0]}_20.json"
            retrieved_results = load_file(retrieved_fp)
            print("Retrieved docs loading done.")
        else:
            retriever = Retriever(args)
            retriever.setup_retriever()
            retrieved_results = retriever.search_document(input_data, args.n_docs)

        id2retrieval = {}
        for item in retrieved_results:
            if args.passages_source == "wikipedia":
                id2retrieval[item["id"]] = [i["title"] + "\n" + i["text"] for i in item["passages"][: args.top_n]]
            elif args.passages_source == "general_knowledge":
                id2retrieval[item["id"]] = [i["passages"].strip() for i in item["passages"][: args.top_n]]
            else:
                raise NotImplementedError(f"Unknown passages_source={args.passages_source}")

        for _, item in enumerate(input_data):
            key = "id" if "id" in item else "index"
            evidences = id2retrieval[item[key]]
            item["paragraph"] = "\n".join(evidences)

        del retrieved_results

    # Normalize golds/instruction fields as original
    for item in input_data:
        if "golds" not in item:
            if "output" in item:
                item["golds"] = item["output"]
            if "answers" in item:
                item["golds"] = item["answers"]
            if "answer" in item:
                item["golds"] = item["answer"]
            if "possible_answers" in item:
                item["golds"] = ast.literal_eval(item["possible_answers"])
            if "answerKey" in item:
                item["golds"] = [item["answerKey"]]
            if "label" in item:
                item["golds"] = item["label"]
            if "Correct_answer" in item:
                item["golds"] = item["Correct_answer"]

        if "instruction" not in item and "question" in item:
            item["instruction"] = item["question"]

        if args.instruction is not None:
            item["instruction"] = args.instruction + "\n\n### Input:\n" + item["instruction"]
        if args.task == "fever" or args.task == "arc_c":
            item["instruction"] = TASK_INST[args.task] + "\n\n### Input:\n" + item["instruction"]

    final_results = []

    # Ollama is request-per-prompt; batching is just looping
    n_full_batches = len(input_data) // args.batch_size
    for idx in tqdm(range(n_full_batches)):
        batch = input_data[idx * args.batch_size : (idx + 1) * args.batch_size]
        processed_batch = [PROMPT_DICT[args.prompt_name].format_map(item) for item in batch]

        preds = []
        for input_instance in processed_batch:
            raw = call_model_ollama(
                input_instance,
                model=args.model_name,
                max_tokens=args.max_new_tokens,
                temperature=0.0,
            )
            pred = postprocess_output(raw, force_choice_letter=args.force_choice_letter)
            preds.append(pred)

        for j, item in enumerate(batch):
            pred = preds[j]
            # Keep consistent with original script (they compute and then overwrite; we keep only pred)
            item["output"] = postprocess_answers_closed(pred, args.task, args.choices)
            item["output"] = pred
            final_results.append(item)

    # remainder
    if len(input_data) % args.batch_size > 0:
        batch = input_data[n_full_batches * args.batch_size :]
        processed_batch = [PROMPT_DICT[args.prompt_name].format_map(item) for item in batch]

        preds = []
        for input_instance in processed_batch:
            raw = call_model_ollama(
                input_instance,
                model=args.model_name,
                max_tokens=args.max_new_tokens,
                temperature=0.0,
            )
            pred = postprocess_output(raw, force_choice_letter=args.force_choice_letter)
            preds.append(pred)

        for j, item in enumerate(batch):
            pred = preds[j]
            item["output"] = postprocess_answers_closed(pred, args.task, args.choices)
            item["output"] = pred
            final_results.append(item)

    # Metric computation (unchanged)
    for item in input_data:
        if args.metric == "em":
            metric_result = metric_max_over_ground_truths(exact_match_score, item["output"], item["golds"])
        elif args.metric == "accuracy":
            metric_result = 1.0 if item["golds"][0] in item["output"] else 0.0
        elif args.metric == "match":
            metric_result = match(item["output"], item["golds"])
        elif args.metric == "exact_match_score":
            metric_result = binary_choice_match(item["output"], item["golds"])
        elif args.metric == "binary_choice_match":
            metric_result = binary_choice_match(item["output"], item["golds"])
        elif args.metric == "multiple_choice_match":
            metric_result = multiple_choice_match(item["output"], item["golds"])
        else:
            raise NotImplementedError(f"Unknown metric={args.metric}")
        item["metric_result"] = metric_result

    all_results = [item["metric_result"] for item in input_data]
    vaild_results = [item for item in all_results if isinstance(item, bool)]
    adjusted_results = [item if isinstance(item, bool) else False for item in all_results]

    print(f"vaild answers/all: {len(vaild_results)}/{len(all_results)}")
    print("overall exact match: {0}".format(np.mean(adjusted_results)))
    print("valid exact match: {0}".format(np.mean(vaild_results)))

    # Save results (unchanged behavior)
    if args.task == "factscore":
        save_file_jsonl(input_data, result_fp)
    else:
        cleaned = [{k: v for k, v in item.items() if k not in ["golds", "instruction"]} for item in input_data]
        save_file_jsonl(cleaned, result_fp)


if __name__ == "__main__":
    main()