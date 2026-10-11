import torch
from torch import nn
from lm_data import PAD, BOS


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, d_model=128,
                 num_layers=2, num_heads=4, feedforward_dim=512,
                 dropout=0.2, context_length=128):
        super().__init__()
        self.context_length = context_length
        # keep 100d input embeddings for the planned word-embedding comparison.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.input_projection = nn.Linear(embedding_dim, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(dropout)

        # create layers separately so they have independent random initialization.
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation='relu',
                batch_first=True,
            )
            self.layers.append(layer)
        self.output = nn.Linear(d_model, vocab_size)

    def token_states(self, ids):
        # Compute contextual states without attending to future or PAD tokens.
        length = ids.shape[1]
        positions = torch.arange(length, device=ids.device)
        word_vectors = self.embedding(ids)
        states = self.input_projection(word_vectors)
        states = states + self.position_embedding(positions)
        states = self.dropout(states)

        # true means blocked: position i cannot attend to any position j > i.
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=ids.device),
            diagonal=1,
        )
        padding_mask = ids.eq(PAD)
        for layer in self.layers:
            states = layer(
                states,
                src_mask=causal_mask,
                src_key_padding_mask=padding_mask,
            )
        return states

    def forward(self, ids):
        states = self.token_states(ids)
        logits = self.output(self.dropout(states))
        # use the same prediction support as all previous language models.
        mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=ids.device)
        mask[PAD] = True
        mask[BOS] = True
        return logits.masked_fill(mask, float('-inf'))
