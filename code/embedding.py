from torch import nn
import torch
class Embedding(nn.Module):
   def __init__(self):
       super(Embedding, self).__init__()
       self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
       self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
       self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
       self.norm = nn.LayerNorm(d_model)

   def forward(self, x, seg):
       seq_len = x.size(1)
       pos = torch.arange(seq_len, dtype=torch.long)
       pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
       embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
       return self.norm(embedding)