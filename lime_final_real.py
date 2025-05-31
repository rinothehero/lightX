#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# Force Transformers to offline mode to prevent Hub requests
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import json
import math
import multiprocessing as mp
import argparse
from datasets import load_dataset

# 워커 프로세스: 각 GPU별로 IG 설명 생성
def ig_worker(gpu_id, texts, labels, out_path, model_path):
    import torch
    import torch.nn.functional as F
    from transformers import BertTokenizerFast, BertForSequenceClassification
    from captum.attr import IntegratedGradients

    # 로컬 경로 확인
    model_path = os.path.abspath(model_path)
    if not os.path.isdir(model_path):
        raise RuntimeError(f"Model path not found or not a directory: {model_path}")

    device = torch.device(f"cuda:{gpu_id}")
    tokenizer = BertTokenizerFast.from_pretrained(model_path, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.eval().to(device)

    ig = IntegratedGradients(model)
    results = []

    for text, true_label in zip(texts, labels):
        encoding = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            padding="max_length",
            max_length=128
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        offsets = encoding["offset_mapping"][0].tolist()

        tokens = [text[start:end] for start, end in offsets if start < end]

        # 모델 예측 및 라벨 결정 (detach before numpy)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_label = int(probs.argmax())

        # baseline 설정 (패딩 토큰 ID로 채운 입력)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id).to(device)

        # IG 계산
        attributions = ig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            target=pred_label,
            additional_forward_args=attention_mask,
            n_steps=50
        )

        # 임베딩 차원 합산 및 패딩 제외
        token_attr = attributions.sum(dim=-1).squeeze(0).cpu().tolist()
        filtered_attr = [token_attr[i] for i, (start, end) in enumerate(offsets) if start < end]

        token_ig = [
            {"token": tok, "attribution": filtered_attr[i]}
            for i, tok in enumerate(tokens)
        ]

        results.append({
            "text": text,
            "true_label": int(true_label),
            "predicted_label": pred_label,
            "token_ig": token_ig,
            "probabilities": probs.tolist()
        })

    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)
    print(f"[GPU {gpu_id}] completed → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute IG attributions across specified GPUs.")
    parser.add_argument(
        '--gpus', type=str, default=None,
        help="Comma-separated list of GPU IDs to use (e.g. '0,2,3'). Defaults to all available GPUs."
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help="Path to the local pretrained BERT model directory."
    )
    args = parser.parse_args()

    import torch

    # GPU 선택
    if args.gpus:
        gpu_list = [int(x) for x in args.gpus.split(',')]
    else:
        gpu_list = list(range(torch.cuda.device_count()))
    if not gpu_list:
        raise RuntimeError("No GPUs specified or available.")

    # 데이터 로드 및 클래스 균형 샘플링
    per_class = 1
    ds = load_dataset("ag_news", split="train")
    counts = {i: 0 for i in range(4)}
    buffer = []
    for ex in ds:
        lbl = ex["label"]
        if counts[lbl] < per_class:
            buffer.append(ex)
            counts[lbl] += 1
        if all(counts[i] == per_class for i in counts):
            break
    texts = [ex["text"] for ex in buffer]
    labels = [ex["label"] for ex in buffer]

    # 프로세스별 작업 분할
    n_workers = len(gpu_list)
    chunk_size = math.ceil(len(texts) / n_workers)

    mp.set_start_method('spawn', force=True)
    procs = []
    for idx, gpu_id in enumerate(gpu_list):
        start = idx * chunk_size
        end = min((idx + 1) * chunk_size, len(texts))
        sub_texts = texts[start:end]
        sub_labels = labels[start:end]
        out_fn = f"ig_gpu{gpu_id}.json"

        p = mp.Process(
            target=ig_worker,
            args=(gpu_id, sub_texts, sub_labels, out_fn, args.model_path)
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # 결과 병합
    merged = []
    for gpu_id in gpu_list:
        fn = f"ig_gpu{gpu_id}.json"
        with open(fn, "r", encoding="utf-8") as fin:
            merged.extend(json.load(fin))

    with open("ig_results_full.json", "w", encoding="utf-8") as fout:
        json.dump(merged, fout, ensure_ascii=False, indent=2)
    print("✅ All GPUs finished → ig_results_full.json")


if __name__ == "__main__":
    main()