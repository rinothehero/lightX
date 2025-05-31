#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP 분류 모델 학습 스크립트 최종판
- 데이터: lime_results_full.json
- 모델: 임베딩 기반 MLP
- Loss: L = α·L_ce + (1-α)·L_xai
  * L_ce: cross entropy
  * L_xai: (1-cos) + (1-ρ)
- Subword 단위 LIME 설명 비교
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from lime.lime_text import LimeTextExplainer
from scipy.stats import spearmanr
import numpy as np
import math

# 하이퍼파라미터
RESULT_JSON = 'lime_results_full.json'
BATCH_SIZE  = 1
MAX_LEN     = 128
EMBED_DIM   = 128
HIDDEN_DIM  = 256
N_CLASSES   = 4
LR          = 1e-3
EPOCHS      = 5
ALPHA       = 0.1
N_SAMPLES   = 500
DEVICE      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"[INFO] Using device: {DEVICE}")

# 데이터셋 정의
class LimeXaiDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len=128):
        data = json.load(open(json_path, 'r', encoding='utf-8'))
        self.texts      = [ex['text'] for ex in data]
        self.labels     = [ex['true_label'] for ex in data]
        self.target_xai = [ex['token_lime'] for ex in data]
        self.tokenizer  = tokenizer
        self.max_len    = max_len

    def __len__(self):
        #print(f"[DEBUG] Dataset size: {len(self.texts)}")
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )
        input_ids      = torch.tensor(enc['input_ids'],      dtype=torch.long)
        attention_mask = torch.tensor(enc['attention_mask'], dtype=torch.long)

        # 1) subword 토큰 문자열 추출
        offsets = enc['offset_mapping']
        tokens  = [ text[s:e] for s, e in offsets if s<e ]

        # 2) JSON에서 토큰별 weight 꺼내기
        #    self.target_xai[idx] 은 list of dicts
        raw_lime = self.target_xai[idx]

        #    weights_list: [[w0…], [w0…], …] 순서대로
        weights_list = [ entry['weights'] for entry in raw_lime ]

        tokens_json = [ entry['token'] for entry in raw_lime ]


        # 3) 토큰 길이 mismatch 방지 (truncate or pad)
        if len(weights_list) > self.max_len:
            weights_list = weights_list[:self.max_len]
        pad_size = self.max_len - len(weights_list)
        if pad_size > 0:
            weights_list.extend([[0.0]*N_CLASSES] * pad_size)

        # 4) tensor 변환
        bert_shap = torch.tensor(weights_list, dtype=torch.float)  # shape [max_len, C]
        label     = torch.tensor(self.labels[idx], dtype=torch.long)

        return text, input_ids, attention_mask, bert_shap, label, tokens_json

# MLP 모델 정의
class MLPWithEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1       = nn.Linear(embed_dim, hidden_dim)
        self.fc2       = nn.Linear(hidden_dim, output_dim)
        self.dropout   = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)              # [B, L, E]
        mask = attention_mask.unsqueeze(-1)          # [B, L, 1]
        summed = (emb * mask).sum(dim=1)            # [B, E]
        counts = mask.sum(dim=1).clamp(min=1)        # [B, 1]
        pooled = summed / counts                    # [B, E]
        x = self.dropout(F.relu(self.fc1(pooled)))  # [B, H]
        logits = self.fc2(x)                        # [B, C]
        return logits

# XAI 손실 함수
def xai_loss(mlp_shap, bert_shap):
    B, L, C = mlp_shap.shape
    cos_losses, spr_losses = [], []
    for i in range(B):
        mask = bert_shap[i].abs().sum(dim=-1) > 0
        L_i = int(mask.sum().item())
        if L_i==0:
            cos_losses.append(torch.tensor(0., device=mlp_shap.device))
            spr_losses.append(torch.tensor(0., device=mlp_shap.device))
            continue

        # slice out real tokens
        m = mlp_shap[i][:L_i]   # [L_i, C]
        b = bert_shap[i][:L_i]
        cos_c = []
        spr_c = []
        for mc, bc in zip(m.unbind(-1), b.unbind(-1)):
            c = F.cosine_similarity(mc.unsqueeze(0), bc.unsqueeze(0), dim=1).item()
            r, _ = spearmanr(mc.cpu().numpy(), bc.cpu().numpy())
            cos_c.append(c)
            spr_c.append(0. if np.isnan(r) else r)
        cos_mean = sum(cos_c)/C
        spr_mean = sum(spr_c)/C

        # normalize each to [0,1]
        cos_losses.append((1 - cos_mean)/2)
        spr_losses.append((1 - spr_mean)/2)

    # 최종 손실: 두 항 평균
    cos_tensor = torch.tensor(cos_losses, device=mlp_shap.device)
    spr_tensor = torch.tensor(spr_losses, device=mlp_shap.device)
    return (cos_tensor + spr_tensor).mean()

# collate_fn 정의
def collate_fn(batch):
    texts, input_ids, attention_mask, bert_shap, labels, tokens_json = zip(*batch)
    return (
        list(texts),
        torch.stack(input_ids),
        torch.stack(attention_mask),
        torch.stack(bert_shap),
        torch.stack(labels),
        list(tokens_json)
    )

# 학습 루프

def main():
    tokenizer = BertTokenizerFast.from_pretrained('../../models/ft_BERT_agnews_full_dataset')
    dataset = LimeXaiDataset(RESULT_JSON, tokenizer, max_len=MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = MLPWithEmbedding(tokenizer.vocab_size, EMBED_DIM, HIDDEN_DIM, N_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ce_loss = nn.CrossEntropyLoss()

    # predict_proba 함수
    def predict_proba(texts):
        enc = tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=MAX_LEN
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(enc['input_ids'], enc['attention_mask'])
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    explainer = LimeTextExplainer(
        class_names=[str(i) for i in range(N_CLASSES)],
        split_expression=lambda txt: [
            txt[s:e] for s, e in tokenizer(
                txt,
                return_offsets_mapping=True,
                truncation=True,
                max_length=MAX_LEN
            )['offset_mapping'] if s < e
        ],
        bow=False
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for texts, input_ids, attention_mask, bert_shap, labels , tokens_json_list in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            bert_shap = bert_shap.to(DEVICE)
            labels = labels.to(DEVICE)

            # → 여기서 바로 한 문장씩 비교해보기
        for i, txt in enumerate(texts):
            # JSON 분할
            print(f"[DEBUG] 문장: {txt}")
            print(f"[DEBUG] JSON tokens ({len(tokens_json_list[i])}): {tokens_json_list[i]}")

            # MLP tokenizer 분할
            offsets = tokenizer(
                txt,
                return_offsets_mapping=True,
                truncation=True,
                max_length=MAX_LEN
            )['offset_mapping']
            toks_mlp = [txt[s:e] for s, e in offsets if s < e]
            print(f"[DEBUG] MLP tokens  ({len(toks_mlp)}): {toks_mlp}")

            # 일치 여부 간단 검사
            if tokens_json_list[i] != toks_mlp:
                print("⚠️ 토큰 불일치! JSON vs MLP tokenizer가 다릅니다.")




            # (a) CrossEntropy
            logits = model(input_ids, attention_mask)
            L_ce = ce_loss(logits, labels)
            # 수정: 정규화
            max_ce = math.log(N_CLASSES)        # = log(C)
            L_ce_norm = L_ce / max_ce           # roughly in [0,1]


            # (b) MLP LIME 설명 획득 (토큰+weight 리스트로 저장 & [MAX_LEN,C] 패딩 → mlp_shap 생성)
            batch_token_lime = []
            batch_shap      = []
            for txt in texts:
                # 1) subword 토큰 문자열 추출
                offsets = tokenizer(
                    txt,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=MAX_LEN
                )['offset_mapping']
                toks = [txt[s:e] for s, e in offsets if s < e]

                # 2) LIME 설명 호출
                explanation = explainer.explain_instance(
                    text_instance=txt,
                    classifier_fn=predict_proba,
                    num_features=len(toks),
                    labels=list(range(N_CLASSES)),
                    num_samples=N_SAMPLES
                )
                amap = explanation.as_map()

                # 3) 위치 기반 raw_weights 초기화
                raw_weights = [[0.0]*N_CLASSES for _ in range(len(toks))]
                for cls_idx, idx_weights in amap.items():
                    for tok_idx, w in idx_weights:
                        if tok_idx < len(toks):
                            raw_weights[tok_idx][cls_idx] = w

                # 4) 토큰 문자열+weights 형태로 저장
                token_lime = [
                    {"token": toks[i], "weights": raw_weights[i]}
                    for i in range(len(toks))
                ]
                batch_token_lime.append(token_lime)
                #print(f"[DEBUG] token_lime: {token_lime[:]}")  # 디버그 출력

                # 5) raw_weights ([L, C]) → [MAX_LEN, C] 패딩
                L_i = len(raw_weights)
                padded = torch.zeros((MAX_LEN, N_CLASSES), device=DEVICE)
                if L_i > 0:
                    padded[:L_i, :] = torch.tensor(raw_weights, device=DEVICE)
                batch_shap.append(padded)
            # 리스트를 스택해서 (B, MAX_LEN, C) 텐서로 변환
            mlp_shap = torch.stack(batch_shap)  # shape [B, MAX_LEN, N_CLASSES]
            
            print(f"[DEBUG] mlp_shap shape: {mlp_shap.shape}")  # 디버그 출력
            print(f"[DEBUG] mlp_shap: {mlp_shap}")  # 디버그 출력
            
            print(f"[DEBUG] bert_shap shape: {bert_shap.shape}")  # 디버그 출력
            print(f"[DEBUG] bert_shap: {bert_shap}")  # 디버그 출력
            


            # (c) XAI 손실
            L_xai = xai_loss(mlp_shap, bert_shap)

            # (d) Combined Loss
            loss = ALPHA * L_ce_norm + (1 - ALPHA) * L_xai
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch}] Cross Entropy Loss: {L_ce.item():.4f}, Normalized: {L_ce_norm.item():.4f}, XAI Loss: {L_xai.item():.4f}")
        print(f"[Epoch {epoch}] Avg Loss: {total_loss / len(loader):.4f}\n")
       

    torch.save(model.state_dict(), 'mlp_xai_model.pt')

if __name__ == '__main__':
    main()