import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data._utils.collate import default_collate
from transformers import BertTokenizerFast


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train MLP on AG News with manual IG XAI imitation loss'
    )
    parser.add_argument('--data_json', type=str, default='ig_results.json',
                        help='Path to IG results JSON file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Local path to pretrained BERT tokenizer')
    parser.add_argument('--alpha1', type=float, required=True,
                        help='alpha when mlp_pred == bert_pred == true_label')
    parser.add_argument('--alpha2', type=float, required=True,
                        help='alpha when mlp_pred == true_label != bert_pred')
    parser.add_argument('--alpha3', type=float, required=True,
                        help='alpha otherwise')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpus', type=str, default='0',
                        help='Comma-separated GPU ids, e.g. "0,1"')
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--n_steps', type=int, default=20,
                        help='Number of IG steps for Riemann sum')
    parser.add_argument('--debug_grad', action='store_true',
                        help='Enable gradient debugging')
    parser.add_argument('--debug_detail', action='store_true',
                        help='Enable detailed per-sample debug')
    return parser.parse_args()


def collate_fn(batch):
    texts = [item.pop('text') for item in batch]
    raw_igs = [item.pop('raw_token_ig') for item in batch]
    batch_collated = default_collate(batch)
    batch_collated['text'] = texts
    batch_collated['raw_token_ig'] = raw_igs
    return batch_collated


class IGDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len, num_classes=4):
        p = Path(json_path)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {json_path}")
        with open(p, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        text = entry['text']
        true_label = entry['true_label']
        bert_pred = entry['predicted_label']
        bert_token_ig = entry['token_ig']

        enc = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        input_ids = enc['input_ids'].squeeze(0)
        mask = enc['attention_mask'].squeeze(0)
        offsets = enc['offset_mapping'].squeeze(0).tolist()

        # Build BERT IG tensor [C, L]
        bert_ig = torch.zeros(self.num_classes, self.max_len)
        for i, (s, e) in enumerate(offsets):
            if mask[i] == 1 and i < len(bert_token_ig):
                atts = bert_token_ig[i]['attributions']
                for c, val in enumerate(atts):
                    bert_ig[c, i] = val

        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': mask,
            'true_label': torch.tensor(true_label, dtype=torch.long),
            'bert_pred': torch.tensor(bert_pred, dtype=torch.long),
            'bert_ig': bert_ig,
            'raw_token_ig': bert_token_ig
        }


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
        masked = embeds * mask
        mean_pooled = masked.sum(1) / mask.sum(1)
        x = F.relu(self.fc1(mean_pooled))
        logits = self.fc2(x)
        return logits


def compute_mlp_ig_all_manual(model, input_ids, mask, device, num_classes, n_steps):
    # Move inputs
    input_ids = input_ids.to(device)
    mask = mask.to(device)
    # Embedding layer
    embed_layer = model.module.embedding if hasattr(model, 'module') else model.embedding
    # Original and baseline embeddings
    inputs_embeds = embed_layer(input_ids)
    baseline_ids = torch.full_like(input_ids, embed_layer.padding_idx)
    baseline_embeds = embed_layer(baseline_ids)
    # Ensure requires_grad for IG
    inputs_embeds.requires_grad_(True)

    # Difference
    diff = inputs_embeds - baseline_embeds
    B, L, E = diff.shape
    mlp_ig_list = []

    # Loop over classes
    for c in range(num_classes):
        total_grad = torch.zeros_like(inputs_embeds)
        # Riemann sum approximating the integral
        for alpha in torch.linspace(0, 1, steps=n_steps, device=device):
            scaled = baseline_embeds + alpha * diff
            scaled.requires_grad_(True)
            logits = model(inputs_embeds=scaled, attention_mask=mask)
            score = logits[:, c].sum()
            # create_graph=True for proper gradient propagation
            grad = torch.autograd.grad(score,
                                       scaled,
                                       create_graph=True)[0]
            total_grad += grad

        # IG attribution: element-wise product
        attr = diff * total_grad / n_steps  # [B, L, E]
        # Sum over embedding dims and apply mask
        attr = attr.sum(-1) * mask          # [B, L]
        mlp_ig_list.append(attr)

    # Stack to [B, C, L]
    mlp_ig = torch.stack(mlp_ig_list, dim=1)
    return mlp_ig


def train_epoch(model, dataloader, optimizer, args, device, tokenizer):
    model.train()
    total_ce = total_xai = total_loss = 0.0
    for batch in dataloader:
        # reset gradients at start of batch
        optimizer.zero_grad()

        # move tensors to device
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        true_labels = batch['true_label'].to(device)
        bert_preds = batch['bert_pred'].to(device)
        bert_igs = batch['bert_ig'].to(device)

        # single forward
        logits = model(input_ids=input_ids, attention_mask=mask)
        ce_per_sample = F.cross_entropy(logits, true_labels, reduction='none')

        # manual IG for all classes
        mlp_ig = compute_mlp_ig_all_manual(
            model, input_ids, mask, device,
            num_classes=logits.size(1), n_steps=args.n_steps
        )
        bert_ig = bert_igs  # already on device

        # compute masked squared error [B, C, L]
        diff2 = (mlp_ig - bert_ig) ** 2
        token_counts = mask.sum(dim=1)
        xai_loss_0 = (diff2[:, 0, :] * mask).sum(dim=1) / token_counts
        xai_loss_1 = (diff2[:, 1, :] * mask).sum(dim=1) / token_counts
        xai_loss_2 = (diff2[:, 2, :] * mask).sum(dim=1) / token_counts
        xai_loss_3 = (diff2[:, 3, :] * mask).sum(dim=1) / token_counts
        xai_per_sample = (xai_loss_0 + xai_loss_1 + xai_loss_2 + xai_loss_3) / 4

        # adaptive alpha
        mlp_preds = logits.argmax(dim=1)
        same = (mlp_preds == bert_preds) & (mlp_preds == true_labels)
        corr = (mlp_preds == true_labels) & (mlp_preds != bert_preds)
        #alpha = torch.where(same, args.alpha1,
        #                    torch.where(corr, args.alpha2, args.alpha3))

        base_alpha = torch.where(
            same, args.alpha1,
            torch.where(corr, args.alpha2, args.alpha3)
        )
        
        # override: if BERT is wrong, force alpha=1 (only CE loss)
        bert_correct = (bert_preds == true_labels)
        alpha = torch.where(bert_correct, base_alpha, torch.ones_like(base_alpha))


        # combined loss
        loss_per_sample = alpha * ce_per_sample + (1 - alpha) * xai_per_sample
        loss = loss_per_sample.mean()

        # backward and step
        loss.backward()
        optimizer.step()

         # Debug prints
        if args.debug_detail:
            for i, txt in enumerate(batch['text']):
                print(f"\n[DEBUG] Sample {i}")
                print(f"Sentence: {txt}")
                print("JSON Tokens + Class IGs:")
                for tok_info in batch['raw_token_ig'][i]:
                    print(f"  {tok_info['token']}: {tok_info['attributions']}")
                print("MLP Tokens + Class IGs:")
                tok_list = tokenizer.convert_ids_to_tokens(input_ids[i])
                for c in range(mlp_ig.size(1)):
                    print(f" Class {c}:")
                    for j, tkn in enumerate(tok_list):
                        if mask[i, j] == 1:
                            print(f"  {tkn}: {mlp_ig[i, c, j].item():.6f}")
                # Alignment check
                bert_tokens = [t['token'] for t in batch['raw_token_ig'][i]]
                mlp_tokens = tok_list
                if len(bert_tokens) != len(mlp_tokens):
                    print(f"  ⚠️ Length mismatch: BERT {len(bert_tokens)} vs MLP {len(mlp_tokens)} tokens")
                print("Token alignment:")
                for idx, (b_tok, m_tok) in enumerate(zip(bert_tokens, mlp_tokens)):
                    status = "OK" if b_tok==m_tok else "❌ MISMATCH"
                    print(f"  pos={idx:2d} BERT='{b_tok}' | MLP='{m_tok}' → {status}")
                print(f"Labels - true: {true_labels[i].item()}, bert_pred: {bert_preds[i].item()}, mlp_pred: {mlp_preds[i].item()}\n")

        # debug gradient flow if requested
        if args.debug_grad:
            print("[DEBUG GRADS]")
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f" No grad for {name}")
                else:
                    print(f" {name}: grad_norm={param.grad.norm().item():.6f}")

        total_ce += ce_per_sample.mean().item()
        total_xai += xai_per_sample.mean().item()
        total_loss += loss.item()

    n = len(dataloader)
    print(f"Training Epoch: CE={total_ce/n:.4f}, XAI={total_xai/n:.4f}, Total={total_loss/n:.4f}")
    print(f"Training Epoch: CE={total_ce/n:.4f}, XAI={total_xai/n:.4f}, Total={total_loss/n:.4f}")


def evaluate(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in dataloader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['true_label'].to(device)
            logits = model(input_ids=ids, attention_mask=mask)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        print("Warning: no validation samples; skipping evaluation.")
    else:
        print(f"Validation Accuracy: {correct/total:.4%}")


def main():
    args = parse_args()

    # GPU setup
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    if any(i >= torch.cuda.device_count() for i in gpu_ids):
        raise ValueError(f"Invalid GPU IDs: {gpu_ids}")
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizerFast.from_pretrained(
        args.model_path,
        local_files_only=True
    )

    dataset = IGDataset(args.data_json, tokenizer, args.max_len)
    total_len = len(dataset)
    if total_len < 2:
        raise ValueError(f"Dataset too small ({total_len}) for train/val split.")
    val_size = max(1, int(0.2 * total_len))
    train_size = total_len - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, collate_fn=collate_fn
    )

    model = MLPClassifier(tokenizer.vocab_size).to(device)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, optimizer, args, device, tokenizer)
    evaluate(model, val_loader, device)

    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state, "models/4class_a_1force_0.0.pt")


if __name__ == '__main__':
    main()
