import torch
from torch import nn
from lm_data import PAD, BOS


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_size=128,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, ids):
        embeddings = self.embedding(ids)
        states, _ = self.lstm(embeddings)
        logits = self.output(self.dropout(states))
        mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
        mask[PAD] = True
        mask[BOS] = True
        return logits.masked_fill(mask, float('-inf'))

    def token_states(self, ids):
        embeddings = self.embedding(ids)
        states, _ = self.lstm(embeddings)
        return states
