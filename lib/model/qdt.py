import torch, logging
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class QuestionGuidedDifferenceTokenizer(nn.Module):
    """
    Concatenate residual maps (R+, R-, Rabs, optional signed conv) -> tokens.
    Add zone tokens (pooled by precomputed masks).
    Cross-attend tiny text embedding to visual tokens, produce gated top-k and heatmap.
    """

    def __init__(
        self, c_img, d_txt=256, k=64, num_rows=3, num_cols=2, use_zone_bias=True
    ):
        super().__init__()
        self.k = k
        self.use_zone_bias = use_zone_bias

        self.txt_to_img = nn.Linear(d_txt, c_img)  # project q to img-dim
        self.tok_proj = nn.Linear(c_img, c_img)  # residual token projection
        self.scale = c_img**-0.5

        # tiny gate over attention scores (per token)
        self.gate = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

        # Zone adjacency bias (fixed grid adjacency)
        self.num_rows, self.num_cols = num_rows, num_cols
        Z = num_rows * num_cols
        adj = torch.zeros(Z, Z)
        for r in range(num_rows):
            for c in range(num_cols):
                i = r * num_cols + c
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < num_rows and 0 <= cc < num_cols:
                        j = rr * num_cols + cc
                        adj[i, j] = 1.0
        self.register_buffer("zone_adj", adj)
        
        self.cached_masks = None  # filled on first forward

    def forward(self, q_vec, r_maps):
        """
        q_vec: (B, d_txt)
        r_maps: dict with keys r_pos, r_neg, r_abs, signed (each B,C,H,W). 'signed' optional.
        """
        r_pos, r_neg, r_abs = r_maps["r_pos"], r_maps["r_neg"], r_maps["r_abs"]
        signed = r_maps.get("signed", None)

        B, C, H, W = r_pos.shape
        if self.cached_masks is None:
            self.cached_masks = self.make_zone_masks(
                H, W, self.num_rows, self.num_cols
            ).to(
                r_pos.device
            )  # (Z,H,W)

        feats = torch.cat([r_pos, r_neg, r_abs], dim=1)  # (B,3C,H,W)
        if signed is not None:
            feats = torch.cat([feats, signed], dim=1)  # (B,4C,H,W)
        
        logger.debug(f"QDT inputs: q_vec={q_vec.shape}, concatenated feature map={feats.shape}")
        # C_all = feats.shape[1]

        tokens = feats.flatten(2).transpose(1, 2)  # (B,HW,C_all)
        tokens = self.tok_proj(tokens)  # (B,HW,C_all)

        # zone tokens: pool per zone mask
        Z = self.cached_masks.shape[0]
        zone_tokens = []
        for z in range(Z):
            m = self.cached_masks[z]  # (H,W)
            m = m.view(1, 1, H, W)
            pooled = (feats * m).sum(dim=(2, 3)) / (m.sum() + 1e-6)  # (B,C_all)
            zone_tokens.append(pooled)
        zone_tokens = torch.stack(zone_tokens, dim=1)  # (B,Z,C_all)

        # concatenate tokens + zone tokens
        mem = torch.cat([tokens, zone_tokens], dim=1)  # (B,HW+Z,C_all)

        # attention from question to memory
        q = self.txt_to_img(q_vec).unsqueeze(1)  # (B,1,C_all)
        att = (q @ mem.transpose(1, 2)) * self.scale  # (B,1,HW+Z)
        att = att.transpose(1, 2)  # (B,HW+Z,1)
        gated = self.gate(att) * att  # gate is between 0 and 1 elementwise
        att_scores = gated.squeeze(-1)  # (B,HW+Z)
        att_weights = F.softmax(att_scores, dim=-1)  # (B,HW+Z)

        # evidence heatmap over spatial tokens only
        heat_spatial = att_weights[:, : H * W].reshape(B, H, W)

        # top-k selection over all tokens (spatial + zones)
        k = min(self.k, mem.shape[1])
        topk_vals, topk_idx = att_weights.topk(k, dim=-1)
        batch = torch.arange(B, device=mem.device)[:, None]
        sel_tokens = mem[batch, topk_idx]  # (B,k,C_all)

        logger.debug(f"QDT outputs: selected_tokens={sel_tokens.shape}, heatmap={heat_spatial.shape}")

        # small sparsity regularizer (encourage the gate to actually reject)
        gate_l1 = gated.abs().mean()

        return sel_tokens, heat_spatial, gate_l1

    def make_zone_masks(self, h, w, num_rows=3, num_cols=2):
        """
        Simple grid-based lung zones: (num_rows x num_cols).
        Returns a tensor (Z, H, W) with binary masks.
        """
        masks = []
        row_edges = torch.linspace(0, h, steps=num_rows + 1).long()
        col_edges = torch.linspace(0, w, steps=num_cols + 1).long()
        for i in range(num_rows):
            for j in range(num_cols):
                m = torch.zeros(h, w, dtype=torch.float32)
                m[row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]] = (
                    1.0
                )
                masks.append(m)
        return torch.stack(masks, dim=0)  # (Z,H,W)


class TinyText(nn.Module):
    # keep this for ablations / classifier baseline
    def __init__(self, vocab_size=4096, d_txt=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_txt)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.pad = 0

    def tokenize(self, questions, max_len=32):
        toks = []
        for q in questions:
            ws = q.strip().lower().split()[:max_len]
            ids = [(hash(w) % (self.emb.num_embeddings - 1)) + 1 for w in ws] or [1]
            ids += [0] * (max_len - len(ids))
            toks.append(ids)
        return torch.tensor(toks, dtype=torch.long)

    def forward(self, token_ids):
        emb = self.emb(token_ids)
        mask = token_ids != self.pad
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (emb * mask.unsqueeze(-1)).sum(dim=1) / denom


class ClinicalBERTText(nn.Module):
    """
    Bio_ClinicalBERT pooled encoder.
    Default: freeze parameters; set fine_tune=True to unfreeze top layers.
    """

    def __init__(
        self,
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        d_txt=768,
        proj_dim=256,
        fine_tune=False,
    ):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.bert = AutoModel.from_pretrained(model_name)
        self.out_dim = d_txt  # usually 768 for BERT-base
        self.proj = (
            nn.Linear(self.out_dim, proj_dim)
            if proj_dim and proj_dim != self.out_dim
            else nn.Identity()
        )

        if not fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

    def tokenize(self, questions, max_len=48):
        # returns a dict of tensors
        enc = self.tok(
            list(questions),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return enc  # {'input_ids','attention_mask','token_type_ids'(maybe)}

    def forward(self, token_batch):
        # token_batch is the dict returned by tokenize(...), already on device
        out = self.bert(**token_batch)
        # Use CLS pooling (or mean pooling of last hidden state)
        cls = out.last_hidden_state[:, 0]  # (B, hidden)
        q = self.proj(cls)  # (B, proj_dim)
        return q
