import torch
import torch.nn as nn

class KmerEncoder(nn.Module):
    def __init__(self, k=3, embed_dim=64):
        super().__init__()
        self.k = k
        vocab = [''.join(p) for p in __import__('itertools').product('ACGT', repeat=k)]
        self.vocab = {s: i for i, s in enumerate(vocab)}
        self.embedding = nn.Embedding(len(self.vocab), embed_dim)
    def forward(self, seq: str):
        kmers = [seq[i:i+self.k] for i in range(len(seq)-self.k+1)]
        ids = torch.tensor([self.vocab.get(k, 0) for k in kmers], dtype=torch.long)
        emb = self.embedding(ids)
        return emb.mean(dim=0)  # (embed_dim,)
