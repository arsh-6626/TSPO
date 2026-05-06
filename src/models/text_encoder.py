
import json
import os

import torch
import torch.nn as nn

from src.config import TEXT_DIM


class SimpleTokenizer:
    def __init__(self, vocab_path=None, max_length=77):
        self.max_length  = max_length
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path) as f:
                self.word_to_idx = json.load(f)

    def encode(self, text):
        if not text:
            return torch.zeros(self.max_length, dtype=torch.long)
        words   = str(text).lower().split()
        indices = [2]
        for w in words[: self.max_length - 2]:
            indices.append(self.word_to_idx.get(w, 1))
        indices.append(3)
        while len(indices) < self.max_length:
            indices.append(0)
        return torch.tensor(indices[: self.max_length], dtype=torch.long)


class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size=50000, embed_dim=512, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc        = nn.Linear(hidden_dim * 2, TEXT_DIM)
        self.norm      = nn.LayerNorm(TEXT_DIM)
        self.dropout   = nn.Dropout(0.1)

    def forward(self, tokens):
        emb             = self.dropout(self.embedding(tokens))
        out, (hidden, _)= self.lstm(emb)
        hidden_cat      = torch.cat([hidden[0], hidden[1]], dim=1)
        text_feat       = self.norm(self.fc(hidden_cat))
        seq_feat        = self.norm(self.fc(out))
        return text_feat, seq_feat, tokens.eq(0)
