import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def normalize_id(x):
    try:
        return int(x)
    except Exception:
        return str(x)


def should_rerun(item, needle: str) -> bool:
    metric_str = str(item.get("metric_result", ""))
    output_str = str(item.get("output", ""))
    return (needle in metric_str) or (needle in output_str)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark_file", required=True, help="Original benchmark JSON file")
    parser.add_argument("--result_file", required=True, help="Original result JSONL file")
    parser.add_argument("--subset_file", required=True, help="Where to save the small benchmark subset (JSON)")
    parser.add_argument("--rerun_result_file", required=True, help="Where main_ollama.py will save rerun results (JSONL)")
    parser.add_argument("--merged_result_file", required=True, help="Final merged result file (JSONL)")

    parser.add_argument(
        "--needle",
        default="No choice id is outputted",
        help='Substring to match in metric_result or output, e.g. "No choice id is outputted", "429 Client Error", "500 Server Error"',
    )

    parser.add_argument("--python_exec", default=sys.executable)
    parser.add_argument("--main_script", required=True, help="Path to main_ollama.py")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--mode", required=True, choices=["vanilla", "retrieval"])
    parser.add_argument("--max_new_tokens", default="128")
    parser.add_argument("--metric", default="multiple_choice_match")
    parser.add_argument("--prompt_name", required=True)
    parser.add_argument("--task", default="qa")
    parser.add_argument("--result_fp_base", required=True)
    parser.add_argument("--batch_size", default="1")

    parser.add_argument("--passages", default=None)
    parser.add_argument("--passages_embeddings", default=None)
    parser.add_argument("--passages_source", default=None)
    parser.add_argument("--retriever_path", default=None)
    parser.add_argument("--load_retrieved_docs", default="False")

    parser.add_argument("--skip_rerun", action="store_true", help="If set, do not call main_ollama.py; only merge using existing rerun_result_file")
    parser.add_argument("--extra_args", nargs="*", default=[])

    args = parser.parse_args()

    benchmark = load_json(args.benchmark_file)
    results = load_jsonl(args.result_file)

    # 1) collect ids that need rerun
    bad_ids = []
    for item in results:
        if should_rerun(item, args.needle):
            bad_ids.append(normalize_id(item.get("id")))

    bad_ids = list(dict.fromkeys(bad_ids))
    print(f"Found {len(bad_ids)} items matching needle: {args.needle!r}")

    if not bad_ids:
        print("Nothing to rerun. Exiting.")
        return

    # 2) create smaller benchmark subset by id
    bad_id_set = set(bad_ids)
    subset = [x for x in benchmark if normalize_id(x.get("id")) in bad_id_set]
    print(f"Subset benchmark size: {len(subset)}")
    save_json(subset, args.subset_file)
    print(f"Saved subset benchmark to: {args.subset_file}")

    # 3) run main_ollama.py on subset
    if not args.skip_rerun:
        cmd = [
            args.python_exec,
            args.main_script,
            "--model_name", args.model_name,
            "--input_file", args.subset_file,
            "--mode", args.mode,
            "--batch_size", args.batch_size,
            "--max_new_tokens", args.max_new_tokens,
            "--metric", args.metric,
            "--prompt_name", args.prompt_name,
            "--task", args.task,
            "--result_fp_base", args.result_fp_base,
        ]

        if args.mode == "retrieval":
            if not all([args.passages, args.passages_embeddings, args.passages_source, args.retriever_path]):
                raise ValueError(
                    "retrieval mode requires --passages --passages_embeddings --passages_source --retriever_path"
                )
            cmd += [
                "--passages", args.passages,
                "--passages_embeddings", args.passages_embeddings,
                "--passages_source", args.passages_source,
                "--retriever_path", args.retriever_path,
                "--load_retrieved_docs", args.load_retrieved_docs,
            ]

        if args.extra_args:
            cmd += args.extra_args

        print("\nRunning rerun command:\n")
        print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
        print()
        subprocess.run(cmd, check=True)

    # 4) locate rerun result file
    rerun_path = Path(args.rerun_result_file)
    if not rerun_path.exists():
        raise FileNotFoundError(f"Expected rerun result file not found: {rerun_path}")

    rerun_results = load_jsonl(str(rerun_path))
    rerun_map = {normalize_id(x.get("id")): x for x in rerun_results}

    # 5) replace original lines by id
    replaced = 0
    merged = []
    for item in results:
        iid = normalize_id(item.get("id"))
        if iid in rerun_map:
            merged.append(rerun_map[iid])
            replaced += 1
        else:
            merged.append(item)

    save_jsonl(merged, args.merged_result_file)
    print(f"Replaced {replaced} items by id.")
    print(f"Saved merged result to: {args.merged_result_file}")


if __name__ == "__main__":
    main()