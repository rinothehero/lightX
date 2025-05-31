from datasets import load_dataset
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MLPWithEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        emb = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        summed = torch.sum(emb * mask, dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        x = self.dropout(F.relu(self.fc1(pooled)))
        logits = self.fc2(x)
        return logits

class AgNewsTestDataset(Dataset):
    def __init__(self, split="test", max_len=128):
        dataset = load_dataset("ag_news", split=split)
        self.texts = dataset["text"]
        self.labels = dataset["label"]
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return input_ids, attention_mask, label

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return acc

def run_test_evaluation():
    BATCH_SIZE = 64
    MAX_LEN = 128
    N_CLASSES = 4
    EMBED_DIM = 128
    HIDDEN_DIM = 256

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = MLPWithEmbedding(tokenizer.vocab_size, EMBED_DIM, HIDDEN_DIM, N_CLASSES)
    model.load_state_dict(torch.load("a/models/mlp_xai_agnews_manualIG_0.8.pt"))
    model.to(device)

    test_dataset = AgNewsTestDataset(max_len=MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    acc = evaluate(model, test_loader, device)
    print(f"✅ Test Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    run_test_evaluation()