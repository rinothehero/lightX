import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'

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
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

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

# manual IG function for one class (reuse from training)
def compute_ig_for_class(model, input_ids, mask, device, target_class, n_steps=50):
    embed_layer = model.module.embedding if hasattr(model, 'module') else model.embedding
    inputs_embeds = embed_layer(input_ids)
    baseline_ids = torch.full_like(input_ids, embed_layer.padding_idx)
    baseline_embeds = embed_layer(baseline_ids)
    diff = inputs_embeds - baseline_embeds
    inputs_embeds.requires_grad_(True)
    total_grad = torch.zeros_like(inputs_embeds)
    for alpha in torch.linspace(0, 1, steps=n_steps, device=device):
        scaled = baseline_embeds + alpha * diff
        scaled.requires_grad_(True)
        logits = model(inputs_embeds=scaled, attention_mask=mask)
        score = logits[:, target_class].sum()
        grad = torch.autograd.grad(score, scaled, create_graph=False)[0]
        total_grad += grad
    attr = diff * total_grad / n_steps
    attr = attr.sum(-1).detach().cpu() * mask.cpu()
    return attr  # shape [B, L]

# extract mode: generate explanations, accuracy, and save mask
# JSONL records will include 'mask' list

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
    ig_bert = IntegratedGradients(lambda ie, mask: bert(inputs_embeds=ie, attention_mask=mask)[0], multiply_by_inputs=False)

    # prepare dataset
    ds = load_dataset('ag_news', split='test')
    loader = DataLoader(ds, batch_size=1)
    out_file = open(args.output, 'w', encoding='utf-8')
    acc = {'orig':0, 'm1':0, 'm2':0}
    total = 0
    for batch in loader:
        text = batch['text'][0]
        enc = tokenizer(text,
                        padding='max_length', truncation=True,
                        max_length=args.max_len, return_tensors='pt')
        ids = enc['input_ids'].to(device)
        mask = enc['attention_mask'].to(device)
        # record mask for filtering pads later
        mask_list = enc['attention_mask'][0].tolist()
        # true label
        true = batch['label'][0].item()
        # forward
        logits_o = bert(ids, mask)[0]
        pred_o = logits_o.argmax(dim=1).item()
        logits1 = mlp1(ids, mask)
        pred1 = logits1.argmax(dim=1).item()
        logits2 = mlp2(ids, mask)
        pred2 = logits2.argmax(dim=1).item()
        for name,p in zip(['orig','m1','m2'], [pred_o, pred1, pred2]):
            if p == true: acc[name]+=1
        total+=1
        # explanations for pred_o
        attr_o = ig_bert.attribute(
            bert.bert.embeddings(ids),
            baselines=bert.bert.embeddings(torch.full_like(ids, tokenizer.pad_token_id)),
            target=pred_o,
            additional_forward_args=(mask,),
            n_steps=args.n_steps
        ).sum(-1).detach().cpu().numpy().tolist()[0]
        # mlp explanations
        attr1 = compute_ig_for_class(mlp1, ids, mask, device, pred_o, n_steps=args.n_steps).tolist()[0]
        attr2 = compute_ig_for_class(mlp2, ids, mask, device, pred_o, n_steps=args.n_steps).tolist()[0]
        # write
        record = {
            'text': text,
            'pred': pred_o,
            'mask': mask_list,
            'orig_attr': attr_o,
            'm1_attr': attr1,
            'm2_attr': attr2
        }
        out_file.write(json.dumps(record)+'\n')
    out_file.close()
    print(f"Acc orig: {acc['orig']/total:.4%}, m1: {acc['m1']/total:.4%}, m2: {acc['m2']/total:.4%}")

# plot mode: read file and plot CDFs, exclude pads using saved mask

def run_plot(args):
    data = [json.loads(line) for line in open(args.input)]
    sims1 = []
    sims2 = []
    cors1 = []
    cors2 = []
    for rec in data:
        o = np.array(rec['orig_attr'])
        a1 = np.array(rec['m1_attr'])
        a2 = np.array(rec['m2_attr'])
        mask = np.array(rec['mask'], dtype=bool)
        # filter out pad positions
        o_f = o[mask]
        a1_f = a1[mask]
        a2_f = a2[mask]
        # cosine similarity
        sims1.append(cosine_similarity(o_f.reshape(1,-1), a1_f.reshape(1,-1))[0,0])
        sims2.append(cosine_similarity(o_f.reshape(1,-1), a2_f.reshape(1,-1))[0,0])
        # spearman
        cors1.append(spearmanr(o_f, a1_f).correlation)
        cors2.append(spearmanr(o_f, a2_f).correlation)
    # sort for CDF
    sims1 = np.sort(sims1); sims2 = np.sort(sims2)
    cors1 = np.sort(cors1); cors2 = np.sort(cors2)
    p = np.arange(1, len(sims1)+1)/len(sims1)
    # plot
    plt.figure()
    plt.plot(sims1, p)
    plt.plot(sims2, p)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('CDF')
    plt.legend(['m1 vs orig','m2 vs orig'])
    plt.show()

    plt.figure()
    plt.plot(cors1, p)
    plt.plot(cors2, p)
    plt.xlabel('Spearman Correlation')
    plt.ylabel('CDF')
    plt.legend(['m1 vs orig','m2 vs orig'])
    plt.show()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['extract','plot'])
    p.add_argument('--orig', type=str, help='path to orig BERT dir')
    p.add_argument('--m1', type=str, help='path to model1 pt')
    p.add_argument('--m2', type=str, help='path to model2 pt')
    p.add_argument('--output', type=str, default='explanations.jsonl')
    p.add_argument('--input', type=str, help='plot input file')
    p.add_argument('--gpus', type=str, default='0')
    p.add_argument('--max_len', type=int, default=128)
    p.add_argument('--n_steps', type=int, default=50)
    args = p.parse_args()
    if args.mode == 'extract':
        run_extract(args)
    else:
        run_plot(args)
