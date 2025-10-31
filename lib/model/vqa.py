from torch import nn
import torch
from .drs import DirectionalResidualStack
from .mrm import MaskedResidualModel
from .qdt import ClinicalBERTText, QuestionGuidedDifferenceTokenizer, TinyText
from .heads import TinyTransformerDecoder


class DiffVQAModel(nn.Module):
    def __init__(
        self,
        backbone="resnet50",
        text_encoder="tiny",
        text_model_name="emilyalsentzer/Bio_ClinicalBERT",
        text_dim=768,
        text_proj_dim=256,
        text_finetune=False,
        topk=64,
        num_rows=3,
        num_cols=2,
        num_classes=1000,
        max_ans_len=32
    ):
        super().__init__()

        # Vision encoder (DRS+)
        self.drs = DirectionalResidualStack(backbone_name=backbone)
        C = self.drs.out_channels
        c_all = C * 4  # [R+, R-, Rabs, signed]

        # Text encoder
        if text_encoder == "clinicalbert":
            self.text = ClinicalBERTText(
                model_name=text_model_name,
                d_txt=text_dim,
                proj_dim=text_proj_dim,
                fine_tune=text_finetune,
            )
            self.uses_hf = True
            q_dim = text_proj_dim
        else:
            self.text = TinyText(d_txt=text_proj_dim)
            self.uses_hf = False
            q_dim = text_proj_dim

        # QDT+ and MaskedResidualModel
        self.qdt = QuestionGuidedDifferenceTokenizer(
            c_img=c_all, d_txt=q_dim, k=topk, num_rows=num_rows, num_cols=num_cols
        )
        self.mrm = MaskedResidualModel(c_all=c_all, mask_ratio=0.6)
        
        self.head = TinyTransformerDecoder(
            dim=c_all, vocab_size=num_classes, nlayer=3, nhead=8, max_len=max_ans_len
        )

    def forward(self, img_ref, img_cur, token_batch):
        # Visual residuals
        r = self.drs(img_ref, img_cur)  # dict: r_pos, r_neg, r_abs, signed

        # Question vector
        q_vec = self.text(token_batch)  # (B, q_dim)

        # Token selection + heatmap + gate sparsity
        sel_tokens, heatmap, gate_l1 = self.qdt(
            q_vec, r
        )  # (B,k,c_all), (B,H,W), scalar

        # MaskedResidualModel on residual token maps
        feats_for_mrm = torch.cat(
            [r["r_pos"], r["r_neg"], r["r_abs"], r["signed"]], dim=1
        )
        mrm_out = self.mrm(feats_for_mrm)

        return {
            "sel_tokens": sel_tokens,
            "heatmap": heatmap,
            "gate_l1": gate_l1,
            **mrm_out,
            **r,
        }
