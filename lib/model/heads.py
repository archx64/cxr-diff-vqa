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
            nn.Linear(dim, num_classes),
        )

    def forward(self, sel_tokens, token_kinds=None):
        """
        sel_tokens: (B,k,Dim)
        token_kinds: (B,k) in {0:ref, 1:cur, 2:abs, 3:zone} or None
        """
        B, K, D = sel_tokens.shape
        if token_kinds is not None:
            # add learned IDEs depending on provenance
            ide = torch.zeros(B, K, D, device=sel_tokens.device)
            if (token_kinds == 0).any():
                ide[token_kinds == 0] = self.ide_ref
            if (token_kinds == 1).any():
                ide[token_kinds == 1] = self.ide_cur
            if (token_kinds == 2).any():
                ide[token_kinds == 2] = self.ide_abs
            sel_tokens = sel_tokens + ide
        pooled = sel_tokens.mean(dim=1)
        return self.cls(pooled)


class TinyTransformerDecoder(nn.Module):
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
        self.mem_proj = nn.Linear(dim, dim)  # Projects visual tokens

    def forward(
        self,
        sel_tokens,
        targets=None,
        # token_kinds=None,
        start_token_id=1,
        end_token_id=2,
    ):
        B, _, _ = sel_tokens.shape
        mem = self.mem_proj(sel_tokens)
        # IDE tags can be added here if needed, but we'll skip for simplicity now

        if targets is None:
            # --- INFERENCE PATH (Greedy Decoding) ---
            # Start with the <start> token for all items in the batch
            y = torch.full((B, 1), start_token_id, dtype=torch.long, device=mem.device)

            for _ in range(self.max_len - 1):
                q = self.tok(y) + self.pos[:, : y.size(1)]
                # Create a causal mask to prevent attending to future tokens
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(y.size(1)).to(
                    mem.device
                )
                h = self.dec(q, mem, tgt_mask=tgt_mask)

                # Get the logits for the very last token
                logit = self.out(h[:, -1])
                next_token_id = logit.argmax(-1, keepdim=True)

                # Append the predicted token to the sequence
                y = torch.cat([y, next_token_id], dim=1)

                # Early stop if all sequences have produced an <end> token
                if (y == end_token_id).any(dim=1).all():
                    break

            return None, y  # Return only the generated IDs
        
        else:
            # --- TRAINING PATH (Teacher Forcing) ---
            # Prepare inputs and targets for teacher forcing
            decoder_input = targets[:, :-1]
            decoder_target = targets[:, 1:]

            q = self.tok(decoder_input) + self.pos[:, : decoder_input.size(1)]
            # Create a causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                decoder_input.size(1)
            ).to(mem.device)
            h = self.dec(q, mem, tgt_mask=tgt_mask)

            logits = self.out(h)

            # Calculate loss, ignoring the <pad> token (ID 0)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                decoder_target.reshape(-1),
                ignore_index=0,
            )
            return logits, loss
