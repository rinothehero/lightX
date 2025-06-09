import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizerFast


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train MLP on AG News with cross-entropy only'
    )
    parser.add_argument('--data_json', type=str, default='./json/ig_results_full.json',
                        help='Path to JSON file with text and true_label')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Local path to pretrained BERT tokenizer')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpus', type=str, default='2',
                        help='Comma-separated GPU ids, e.g. "0,1"')
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--debug', action='store_true',
                        help='Enable per-sample debug prints')
    return parser.parse_args()


class JSONDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len):
        p = Path(json_path)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {json_path}")
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.texts = [entry['text'] for entry in data]
        self.labels = [entry['true_label'] for entry in data]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }


class MLPClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        embeds = self.embedding(input_ids)  # [B, L, E]
        mask = attention_mask.unsqueeze(-1)
        pooled = (embeds * mask).sum(1) / mask.sum(1)  # [B, E]
        x = F.relu(self.fc1(pooled))
        logits = self.fc2(x)
        return logits


def train_epoch(model, loader, optimizer, device, debug, tokenizer):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids, mask)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if debug:
            preds = logits.argmax(dim=1)
            for i, txt in enumerate(batch['text']):
                print(f"[DEBUG] "
                      f"Text: {txt}\n"
                      f"  True: {labels[i].item()}, Pred: {preds[i].item()}\n")

    avg = total_loss / len(loader)
    print(f"Train CE Loss: {avg:.4f}")


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids, mask)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Validation Accuracy: {correct/total:.2%}")


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

    dataset = JSONDataset(args.data_json, tokenizer, args.max_len)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = MLPClassifier(tokenizer.vocab_size).to(device)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_epoch(model, train_loader, optimizer, device, args.debug, tokenizer)
    evaluate(model, val_loader, device)

    # save final model
    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state, "models/train_mlp_ce_only_agnews.pt")


if __name__ == '__main__':
    main()