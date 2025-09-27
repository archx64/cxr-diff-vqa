import torch
from torch import nn
from torch.nn import functional as F
import timm


class DirectionalResidualStack(nn.Module):
    """
    Frozen backbone -> 1x1 conv + GroupNorm calibration -> directional residuals.
    Returns R+ (cur-ref, relu), R- (ref-cur, relu), R_abs, and signed residual.
    """

    def __init__(self, backbone_name="resnet50", out_index=-1, freeze_backbone=True):
        super(DirectionalResidualStack, self).__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, features_only=True, out_indices=[out_index]
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        ch = self.backbone.feature_info[-1]["num_chs"]

        self.calib = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.GroupNorm(num_groups=min(32, ch), num_channels=ch),
        )

        # tiny alignment head g([R+,R-,Rabs]) -> signed residual approx
        self.align = nn.Sequential(
            nn.Conv2d(ch * 3, ch, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(ch, ch, 1, bias=False),
        )

        self.out_channels = ch

    @torch.no_grad()
    def encode(self, x):
        return self.backbone(x)[0]  # (B, C, H, W)

    def forward(self, img_ref, img_cur):
        f_ref = self.calib(self.encode(img_ref))
        f_cur = self.calib(self.encode(img_cur))
        signed = f_cur - f_ref
        r_pos = F.relu(signed)
        r_neg = F.relu(-signed)
        rabs = signed.abs()
        return {
            "r_pos": r_pos,
            "r_neg": r_neg,
            "r_abs": rabs,
            "signed": signed,
        }

    def alignment_loss(self, r_pos, r_neg, r_abs, signed):
        combo = torch.cat([r_pos, r_neg, r_abs], dim=1)
        pred = self.align(combo)
        return F.mse_loss(pred, signed)
