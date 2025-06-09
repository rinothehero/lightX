#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from datasets import load_dataset
from captum.attr import IntegratedGradients
import numpy as np

# MLP definition (same as training)
class MLPClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is None:
            embeds = self.embedding(input_ids)
        else:
            embeds = inputs_embeds
        mask = attention_mask.unsqueeze(-1)
        pooled = (embeds * mask).sum(1) / mask.sum(1)
        x = F.relu(self.fc1(pooled))
        logits = self.fc2(x)
        return logits

# manual IG for MLP
def compute_ig_for_class(model, input_ids, mask, device, target_class, n_steps=50):
    embed_layer = model.module.embedding if hasattr(model, 'module') else model.embedding
    inputs_embeds = embed_layer(input_ids)
    baseline_ids = torch.full_like(input_ids, embed_layer.padding_idx)
    baseline_embeds = embed_layer(baseline_ids)
    diff = inputs_embeds - baseline_embeds
    inputs_embeds.requires_grad_(True)
    total_grad = torch.zeros_like(inputs_embeds)
    # accumulate gradients
    for alpha in torch.linspace(0, 1, steps=n_steps, device=device):
        scaled = baseline_embeds + alpha * diff
        scaled.requires_grad_(True)
        logits = model(inputs_embeds=scaled, attention_mask=mask)
        score = logits[:, target_class].sum()
        grad = torch.autograd.grad(score, scaled, create_graph=False)[0]
        total_grad += grad
    attr = diff * total_grad / n_steps
    attr = attr.sum(-1).detach().cpu() * mask.cpu()
    return attr  # [B, L]

# extract mode: separate orig and mlp outputs

def run_extract(args):
    device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(args.orig, local_files_only=True)
    # load models
    bert = BertForSequenceClassification.from_pretrained(args.orig, local_files_only=True).to(device).eval()
    mlp1 = MLPClassifier(tokenizer.vocab_size).to(device)
    mlp2 = MLPClassifier(tokenizer.vocab_size).to(device)
    mlp1.load_state_dict(torch.load(args.m1, map_location=device))
    mlp2.load_state_dict(torch.load(args.m2, map_location=device))
    mlp1.eval(); mlp2.eval()
    # captum for BERT
    ig_bert = IntegratedGradients(
        lambda ie, mask: bert(inputs_embeds=ie, attention_mask=mask)[0],
        multiply_by_inputs=False
    )

    # prepare dataset
    ds = load_dataset('ag_news', split='test')
    loader = DataLoader(ds, batch_size=1)

    # set up output files
    orig_out = args.orig_output
    mlp_out = args.mlp_output
    orig_exists = Path(orig_out).is_file()
    if orig_exists:
        # 1) 기존 JSONL 로드: 순서가 데이터셋[0]부터 테스트셋 순서와 일치한다고 가정
        with open(orig_out, 'r', encoding='utf-8') as f:
            orig_records = [json.loads(line) for line in f]
        orig_preds = [rec['pred'] for rec in orig_records]
    else:
        # 최초 실행 시에만 설명 생성용 파일 열기
        orig_file = open(orig_out, 'w', encoding='utf-8')
    mlp_file = open(mlp_out, 'w', encoding='utf-8')

    acc = {'orig':0, 'm1':0, 'm2':0}
    total = 0

    for batch in loader:
        text = batch['text'][0]
        enc = tokenizer(
            text,
            padding='max_length', truncation=True,
            max_length=args.max_len, return_tensors='pt'
        )
        ids = enc['input_ids'].to(device)
        mask = enc['attention_mask'].to(device)
        mask_list = enc['attention_mask'][0].tolist()
        true = batch['label'][0].item()
        # predictions
        orig_exists = Path(orig_out).is_file()
    if orig_exists:
        # 1) 기존 JSONL 로드: 순서가 데이터셋[0]부터 테스트셋 순서와 일치한다고 가정
        with open(orig_out, 'r', encoding='utf-8') as f:
            orig_records = [json.loads(line) for line in f]
        orig_preds = [rec['pred'] for rec in orig_records]

        #total += 1
    else:
        # 최초 실행 시에만 설명 생성용 파일 열기
        orig_file = open(orig_out, 'w', encoding='utf-8')


        logits1 = mlp1(ids, mask); pred1 = logits1.argmax(dim=1).item()
        logits2 = mlp2(ids, mask); pred2 = logits2.argmax(dim=1).item()
        for name, p in zip(['orig','m1','m2'], [pred_o, pred1, pred2]):
            if p == true: acc[name] += 1
        total += 1

        # orig IG (only if file didn't exist)
        if not orig_exists:
            attr_o = ig_bert.attribute(
                bert.bert.embeddings(ids),
                baselines=bert.bert.embeddings(torch.full_like(ids, tokenizer.pad_token_id)),
                target=pred_o,
                additional_forward_args=(mask,),
                n_steps=args.n_steps
            ).sum(-1).detach().cpu().numpy().tolist()[0]
            rec_o = {
                'text': text,
                'pred': pred_o,
                'mask': mask_list,
                'orig_attr': attr_o
            }
            orig_file.write(json.dumps(rec_o) + '\n')
    #total += 1
        # mlp IG
        attr1 = compute_ig_for_class(mlp1, ids, mask, device, orig_preds, n_steps=args.n_steps).tolist()[0]
        attr2 = compute_ig_for_class(mlp2, ids, mask, device, orig_preds, n_steps=args.n_steps).tolist()[0]
        rec_mlp = {
            'text': text,
            'pred': orig_preds,
            'mask': mask_list,
            'm1_attr': attr1,
            'm2_attr': attr2
        }
        mlp_file.write(json.dumps(rec_mlp) + '\n')


    # close and report
    if not orig_exists:
        orig_file.close()
        print(f"Generated orig explanations at {orig_out}")
    else:
        print(f"Skipped orig extraction; found existing {orig_out}")
    mlp_file.close()
    print(f"Generated MLP explanations at {mlp_out}")
    print(f"Acc orig: {acc['orig']/total:.4%}, m1: {acc['m1']/total:.4%}, m2: {acc['m2']/total:.4%}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--orig', required=True, type=str, help='path to orig BERT dir')
    p.add_argument('--m1', required=True, type=str, help='path to model1 pt')
    p.add_argument('--m2', required=True, type=str, help='path to model2 pt')
    p.add_argument('--orig-output', type=str, default='orig_explanations.jsonl',
                   help='output file for orig model')
    p.add_argument('--mlp-output', type=str, default='mlp_explanations.jsonl',
                   help='output file for MLP models')
    p.add_argument('--gpus', type=str, default='0')
    p.add_argument('--max_len', type=int, default=128)
    p.add_argument('--n_steps', type=int, default=50)
    args = p.parse_args()
    run_extract(args)
