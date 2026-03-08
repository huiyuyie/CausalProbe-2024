# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import pickle

import torch

import src.contriever
import src.data
import src.normalize_text


def embed_passages(args, passages, model, tokenizer, device: str):
    total = 0
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []

    with torch.no_grad():
        for k, p in enumerate(passages):
            batch_ids.append(p["id"])

            # Your passages jsonl uses {"id": ..., "Answer": "..."}
            if args.no_title or ("title" not in p):
                text = p["Answer"]
            else:
                # Keep compatibility if some passages have title/text fields
                text = (p.get("title", "") + " " + p.get("text", "")).strip()

            if args.lowercase:
                text = text.lower()
            if args.normalize_text:
                text = src.normalize_text.normalize(text)

            batch_text.append(text)

            if len(batch_text) == args.per_gpu_batch_size or k == len(passages) - 1:
                encoded_batch = tokenizer.batch_encode_plus(
                    batch_text,
                    return_tensors="pt",
                    max_length=args.passage_maxlength,
                    padding=True,
                    truncation=True,
                )

                # CPU-only
                encoded_batch = {kk: vv.to(device) for kk, vv in encoded_batch.items()}

                embeddings = model(**encoded_batch)  # (bs, dim)
                embeddings = embeddings.cpu()

                total += len(batch_ids)
                allids.extend(batch_ids)
                allembeddings.append(embeddings)

                batch_text = []
                batch_ids = []

                if k % 100000 == 0 and k > 0:
                    print(f"Encoded passages {total}", flush=True)

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings


def main(args):
    # Load retriever (Contriever)
    model, tokenizer, _ = src.contriever.load_retriever(args.model_name_or_path)
    print(f"Model loaded from {args.model_name_or_path}.", flush=True)

    # CPU-only
    device = "cpu"
    model.eval()
    model = model.to(device)

    # Force fp32 on CPU (ignore --no_fp16 flag behavior)
    # Many CPU ops don't support fp16 well; keep it simple & stable.
    # So we do NOT call model.half() here.

    passages = src.data.load_passages(args.passages)

    shard_size = len(passages) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    if args.shard_id == args.num_shards - 1:
        end_idx = len(passages)

    passages = passages[start_idx:end_idx]
    print(
        f"Embedding generation for {len(passages)} passages from idx {start_idx} to {end_idx}.",
        flush=True,
    )

    allids, allembeddings = embed_passages(args, passages, model, tokenizer, device=device)

    os.makedirs(args.output_dir, exist_ok=True)
    save_file = os.path.join(args.output_dir, args.prefix + f"_{args.shard_id:02d}")

    print(f"Saving {len(allids)} passage embeddings to {save_file}.", flush=True)
    with open(save_file, mode="wb") as f:
        pickle.dump((allids, allembeddings), f)

    print(f"Total passages processed {len(allids)}. Written to {save_file}.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--passages", type=str, default=None, help="Path to passages (jsonl/tsv supported by src.data)")
    parser.add_argument("--output_dir", type=str, default="wikipedia_embeddings", help="dir path to save embeddings")
    parser.add_argument("--prefix", type=str, default="passages", help="prefix path to save embeddings")
    parser.add_argument("--shard_id", type=int, default=0, help="Id of the current shard")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--per_gpu_batch_size", type=int, default=256, help="Batch size for encoder forward pass")
    parser.add_argument("--passage_maxlength", type=int, default=256, help="Maximum number of tokens in a passage")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Retriever model name/path")

    # Keep flags for compatibility
    parser.add_argument("--no_fp16", action="store_true", help="Ignored on CPU (always fp32).")
    parser.add_argument("--no_title", action="store_true", help="title not added to the passage body")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text before encoding")

    args = parser.parse_args()

    # IMPORTANT: CPU-only => DO NOT init distributed / slurm
    # src.slurm.init_distributed_mode(args)

    main(args)