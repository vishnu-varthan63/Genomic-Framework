import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=64, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, seq_emb, mut_emb):
        # seq_emb: (batch, 1, embed_dim), mut_emb: (batch, n_mut, embed_dim)
        # Concatenate sequence token with mutations as queries
        # Use sequence as query, mutations as key/value
        # For demo we let sequence attend to mutations
        query = seq_emb
        key = value = mut_emb
        attn_output, _ = self.attn(query, key, value)
        return self.norm(attn_output.squeeze(1))
