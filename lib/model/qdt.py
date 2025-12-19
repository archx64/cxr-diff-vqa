import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)

class QuestionGuidedDifferenceTokenizer(nn.Module):
    def __init__(
        self, c_img, d_txt=256, k=64, num_rows=3, num_cols=2, use_zone_bias=True
    ):
        super().__init__()
        self.k = k
        self.use_zone_bias = use_zone_bias 

        # Projects text to visual dimension
        self.txt_to_img = nn.Linear(d_txt, c_img)
        
        # Project visual tokens
        self.tok_proj = nn.Linear(c_img, c_img) 
        
        self.scale = c_img**-0.5
        self.gate = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

        self.num_rows, self.num_cols = num_rows, num_cols
        self.cached_masks = None 

    def forward(self, q_vec, feats):
        """
        Modified forward pass for Ablation Support.
        
        Args:
            q_vec: Question vector [Batch, d_txt]
            feats: Pre-processed feature tensor [Batch, C_all, H, W]
                   (This can be just R_abs, or the full stack)
        """
        B, C_all, H, W = feats.shape

        # --- 1. Zone Mask Creation ---
        if self.cached_masks is None:
            self.cached_masks = self.make_zone_masks(
                H, W, self.num_rows, self.num_cols
            ).to(feats.device)

        # --- 2. Create Visual Tokens ---
        
        # A. Create "Patch Tokens"
        # Flatten spatial dims: (B, C, H, W) -> (B, H*W, C)
        tokens = feats.flatten(2).transpose(1, 2)
        tokens = self.tok_proj(tokens)  

        # B. Create "Zone Tokens"
        Z = self.cached_masks.shape[0]
        zone_tokens = []
        for z in range(Z):
            m = self.cached_masks[z].view(1, 1, H, W)
            # Global Average Pooling within the zone mask
            pooled = (feats * m).sum(dim=(2, 3)) / (m.sum() + 1e-6)
            zone_tokens.append(pooled)
        zone_tokens = torch.stack(zone_tokens, dim=1) 
        zone_tokens = self.tok_proj(zone_tokens)

        # C. Combine into Memory
        mem = torch.cat([tokens, zone_tokens], dim=1)  

        # --- 3. Cross-Attention ---
        q = self.txt_to_img(q_vec).unsqueeze(1)  
        
        # Dot product attention
        att = (q @ mem.transpose(1, 2)) * self.scale  
        att = att.transpose(1, 2)
        
        # --- 4. Gated Selection ---
        gated = self.gate(att) * att  
        att_scores = gated.squeeze(-1)  
        att_weights = F.softmax(att_scores, dim=-1)

        # --- 5. Extract Outputs ---
        # Heatmap for visualization (only the patch tokens)
        heat_spatial = att_weights[:, : H * W].reshape(B, H, W)

        # Top-K Selection
        k = min(self.k, mem.shape[1])
        topk_vals, topk_idx = att_weights.topk(k, dim=-1)
        
        batch = torch.arange(B, device=mem.device)[:, None]
        sel_tokens = mem[batch, topk_idx]  
        
        gate_l1 = gated.abs().mean()

        return sel_tokens, heat_spatial, gate_l1

    def make_zone_masks(self, h, w, num_rows=3, num_cols=2):
        masks = []
        row_edges = torch.linspace(0, h, steps=num_rows + 1).long()
        col_edges = torch.linspace(0, w, steps=num_cols + 1).long()
        
        for i in range(num_rows):
            for j in range(num_cols):
                m = torch.zeros(h, w, dtype=torch.float32)
                m[row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]] = 1.0
                masks.append(m)
        return torch.stack(masks, dim=0)


class ClinicalBERTText(nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", d_txt=768,
                 proj_dim=256, fine_tune=False):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.bert = AutoModel.from_pretrained(model_name)
        self.out_dim = d_txt
        self.proj = nn.Linear(self.out_dim, proj_dim) if proj_dim and proj_dim != self.out_dim else nn.Identity()

        if not fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

    def tokenize(self, questions, max_len=48):
        enc = self.tok(
            list(questions),
            padding=True,
            truncation=True, 
            max_length=max_len,
            return_tensors="pt"
        )
        return enc 

    def forward(self, token_batch):
        out = self.bert(**token_batch)
        cls = out.last_hidden_state[:, 0] 
        q = self.proj(cls)                
        return q