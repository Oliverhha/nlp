# shared document boundaries and next-token targets for Part I models
from pathlib import Path

PAD, UNK, BOS, EOS = 0, 1, 2, 3


def load_vocab(data_dir):
    words = (Path(data_dir) / 'vocab.txt').read_text(encoding='utf-8').splitlines()
    if words[:4] != ['<pad>', '<unk>', '<bos>', '<eos>'] or len(set(words)) != len(words):
        raise ValueError('Invalid vocabulary/special IDs.')
    return words


def documents(data_dir, split, vocab_size, limit=None):
    paths = sorted((Path(data_dir) / split).glob('*.txt'))
    if not paths:
        raise ValueError(f'No encoded TXT files for {split}.')
    if limit is not None:
        paths = paths[:limit]
    for path in paths:
        ids = [int(s) for s in path.read_text(encoding='utf-8').split()]
        if len(ids) < 3 or ids[0] != BOS or ids[-1] != EOS:
            raise ValueError(f'Invalid BOS/EOS or empty review: {path}')
        if any(i < 0 or i >= vocab_size for i in ids):
            raise ValueError(f'Out-of-range token ID: {path}')
        if PAD in ids or BOS in ids[1:] or EOS in ids[:-1]:
            raise ValueError(f'Unexpected special ID inside review: {path}')
        yield ids


def chunks(ids, context_length=128):
    """each target appears exactly once; adjacent chunks share one boundary ID.

    a chunk's first input is the preceding chunk's last target. No hidden state
    carries over between chunks. BOS is input-only; EOS is a scored target.
    """
    if context_length < 1:
        raise ValueError('context_length must be positive.')
    for start in range(0, len(ids) - 1, context_length):
        stop = min(start + context_length, len(ids) - 1)
        yield ids[start:stop], ids[start + 1:stop + 1]


def events(ids, context_length=128):
    """yield (previous_previous, previous, target), respecting shared chunks."""
    for inputs, targets in chunks(ids, context_length):
        for position, target in enumerate(targets):
            yield (inputs[position - 1] if position else None), inputs[position], target


class ChunkDataset:
    """torch-compatible dataset; compact document storage, no expanded logits."""
    def __init__(self, data_dir, split, context_length=128, limit=None):
        from array import array
        self.context_length = context_length
        vocab_size = len(load_vocab(data_dir))
        self.docs = [array('i', ids) for ids in documents(data_dir, split, vocab_size, limit)]
        self.index = [(d, start) for d, ids in enumerate(self.docs)
                      for start in range(0, len(ids)-1, context_length)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        import torch
        d, start = self.index[index]
        ids = self.docs[d]
        stop = min(start + self.context_length, len(ids)-1)
        return (torch.tensor(ids[start:stop], dtype=torch.long),
                torch.tensor(ids[start+1:stop+1], dtype=torch.long))


def collate_chunks(batch):
    from torch.nn.utils.rnn import pad_sequence
    inputs, targets = zip(*batch)
    return (pad_sequence(inputs, batch_first=True, padding_value=PAD),
            pad_sequence(targets, batch_first=True, padding_value=PAD))
