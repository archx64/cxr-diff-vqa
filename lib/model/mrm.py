import torch
from torch import nn
from torch.nn import functional as F


class MaskedResidualModel(nn.Module):
    def __init__(self, c_all, mask_ratio=0.6):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.pre = nn.Sequential(
            nn.Conv2d(c_all, c_all, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_all, c_all, 3, padding=1),
        )
        self.enc = nn.Linear(c_all, c_all)
        self.dec = nn.Linear(c_all, c_all)
        self.mask_token = nn.Parameter(torch.randn(1, 1, c_all))

    def forward(self, feats):  # feats: (B,C,H,W)
        B, C, _, _ = feats.shape # B, C, H, W
        x = self.pre(feats)
        patches = x.flatten(2).transpose(1, 2)  # (B,HW,C)

        device = patches.device
        N = patches.shape[1]
        num_mask = int(self.mask_ratio * N)
        rand = torch.rand(B, N, device=device).argsort(-1)
        masked_idx, unmasked_idx = rand[:, :num_mask], rand[:, num_mask:]
        b = torch.arange(B, device=device)[:, None]

        enc_all = self.enc(patches)
        enc_unmasked = enc_all[b, unmasked_idx]
        full = self.mask_token.expand(B, N, C).clone()
        # full[b, unmasked_idx] = enc_unmasked

        full[b, unmasked_idx] = enc_unmasked.to(full.dtype)

        recon_all = self.dec(full)
        recon_masked = recon_all[b, masked_idx]
        orig_masked = patches[b, masked_idx]

        loss = F.mse_loss(recon_masked, orig_masked)
        return {
            "loss_mrm": loss,
            "patches": patches,
            "recon_all": recon_all,
            "masked_idx": masked_idx,
            "unmasked_idx": unmasked_idx,
        }
