import torch
from torch import nn
from torch.nn import functional as F


class IDEClassifier(nn.Module):
    """Simple closed-vocab classifier on pooled tokens; includes IDE embeddings."""
    def __init__(self, dim, num_classes):
        super().__init__()
        self.ide_ref = nn.Parameter(torch.randn(1, 1, dim))
        self.ide_cur = nn.Parameter(torch.randn(1, 1, dim))
        self.ide_abs = nn.Parameter(torch.randn(1, 1, dim))
        self.cls = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_classes)
        )

    def forward(self, sel_tokens, token_kinds=None):
        """
        sel_tokens: (B,k,Dim)
        token_kinds: (B,k) in {0:ref, 1:cur, 2:abs, 3:zone} or None
        """
        B,K,D = sel_tokens.shape
        if token_kinds is not None:
            # add learned IDEs depending on provenance
            ide = torch.zeros(B,K,D, device=sel_tokens.device)
            if (token_kinds==0).any(): ide[token_kinds==0] = self.ide_ref
            if (token_kinds==1).any(): ide[token_kinds==1] = self.ide_cur
            if (token_kinds==2).any(): ide[token_kinds==2] = self.ide_abs
            sel_tokens = sel_tokens + ide
        pooled = sel_tokens.mean(dim=1)
        return self.cls(pooled)


class TinyTransformerDecoder(nn.Module):
    """
    Optional small seq decoder (for short phrases). You can switch heads in train.py.
    """

    def __init__(self, dim=768, vocab_size=5000, nlayer=3, nhead=8, max_len=32):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim))
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4, batch_first=True
        )
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=nlayer)
        self.out = nn.Linear(dim, vocab_size)
        self.mem_proj = nn.Linear(dim, dim)

        # IDE tags (ref/cur/abs/zone) merged into memory
        self.ide_ref = nn.Parameter(torch.randn(1, 1, dim))
        self.ide_cur = nn.Parameter(torch.randn(1, 1, dim))
        self.ide_abs = nn.Parameter(torch.randn(1, 1, dim))
        self.ide_zone = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, sel_tokens, targets=None, token_kinds=None):
        B, K, D = sel_tokens.shape
        mem = self.mem_proj(sel_tokens)
        if token_kinds is not None:
            ide = torch.zeros_like(mem)
            ide[token_kinds == 0] = self.ide_ref
            ide[token_kinds == 1] = self.ide_cur
            ide[token_kinds == 2] = self.ide_abs
            ide[token_kinds == 3] = self.ide_zone
            mem = mem + ide

        if targets is None:
            # greedy decode for demo
            y = torch.zeros(B, self.max_len, dtype=torch.long, device=mem.device)
            logits_seq = []
            for t in range(self.max_len):
                q = self.tok(y) + self.pos[:, : y.size(1)]
                h = self.dec(q, mem)
                logit = self.out(h[:, -1])
                logits_seq.append(logit.unsqueeze(1))
                y[:, t] = logit.argmax(-1)
            return torch.cat(logits_seq, dim=1), y
        else:
            q = self.tok(targets) + self.pos[:, : targets.size(1)]
            h = self.dec(q, mem)
            logits = self.out(h)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0
            )
            return logits, loss
