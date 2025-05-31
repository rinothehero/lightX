#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# Prevent any Hub calls
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import json
import math
import multiprocessing as mp
import argparse
from datasets import load_dataset

# 워커 프로세스: 각 GPU별로 IG 설명 생성 (모든 클래스 대상, no filtering)
def ig_worker(gpu_id, texts, labels, out_path, model_path, class_names):
    import torch
    import torch.nn.functional as F
    from transformers import BertTokenizerFast, BertForSequenceClassification
    from captum.attr import IntegratedGradients

    # 로컬 모델 경로 체크
    model_path = os.path.abspath(model_path)
    if not os.path.isdir(model_path):
        raise RuntimeError(f"Model path not found: {model_path}")

    device = torch.device(f"cuda:{gpu_id}")
    tokenizer = BertTokenizerFast.from_pretrained(model_path, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.eval().to(device)

    # 임베딩 레이어 설정
    embed_layer = model.bert.embeddings.word_embeddings
    # IG 설정: forward takes inputs_embeds and attention_mask
    ig = IntegratedGradients(
        forward_func=lambda inputs_embeds, mask: model(
            inputs_embeds=inputs_embeds,
            attention_mask=mask
        ).logits
    )

    num_classes = len(class_names)
    results = []

    for text, true_label in zip(texts, labels):
        # 토크나이징 (no offsets)
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # 토큰 문자열 (includes [CLS], [SEP], pads)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # 입력 임베딩
        inputs_embeds = embed_layer(input_ids)

        # 모델 예측
        logits = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_label = int(probs.argmax())

        # 베이스라인 임베딩 (pad tokens)
        pad_id = tokenizer.pad_token_id
        baseline_ids = torch.full_like(input_ids, pad_id)
        baseline_embeds = embed_layer(baseline_ids)

        # 클래스별 IG 계산
        attributions_per_class = []
        for cls_idx in range(num_classes):
            attr = ig.attribute(
                inputs_embeds,
                baselines=baseline_embeds,
                target=cls_idx,
                additional_forward_args=attention_mask,
                n_steps=50
            )
            # sum over embedding dimension
            token_attr = attr.sum(dim=-1).squeeze(0).cpu().tolist()
            attributions_per_class.append(token_attr)

        # 토큰별 클래스별 기여도 결합
        token_ig = []
        for i, tok in enumerate(tokens):
            weights = [attributions_per_class[c][i] for c in range(num_classes)]
            token_ig.append({"token": tok, "attributions": weights})

        results.append({
            "text": text,
            "true_label": int(true_label),
            "predicted_label": pred_label,
            "token_ig": token_ig,
            "probabilities": probs.tolist(),
            "class_names": class_names
        })

    # 결과 저장
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)
    print(f"[GPU {gpu_id}] completed → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-class IG attributions across specified GPUs, including special tokens."
    )
    parser.add_argument(
        '--gpus', type=str, default=None,
        help="GPU IDs as comma-separated list, e.g. '0,2'. Defaults to all available GPUs."
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help="Local path to pretrained BERT model directory."
    )
    args = parser.parse_args()

    import torch
    # GPU 리스트 결정
    if args.gpus:
        gpu_list = [int(x) for x in args.gpus.split(',')]
    else:
        gpu_list = list(range(torch.cuda.device_count()))
    if not gpu_list:
        raise RuntimeError("No GPUs specified or available.")

    # 클래스명 정의
    class_names = ["World", "Sports", "Business", "Sci/Tech"]

    # AG News 샘플 균형 데이터로드
    per_class = 2500
    ds = load_dataset("ag_news", split="train")
    counts = {i: 0 for i in range(len(class_names))}
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

    # GPU별 워크로드 분할
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
            args=(gpu_id, sub_texts, sub_labels, out_fn, args.model_path, class_names)
        )
        p.start()
        procs.append(p)

    # 완료 대기
    for p in procs:
        p.join()

    # GPU 결과 병합
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
