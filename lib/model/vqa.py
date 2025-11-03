from torch import nn
import torch
from .drs import DirectionalResidualStack
from .mrm import MaskedResidualModel
from .qdt import ClinicalBERTText, QuestionGuidedDifferenceTokenizer, TinyText
from .heads import TinyTransformerDecoder


class DiffVQAModel(nn.Module):
    """
    This is the main DRIFT-VQA model. It initializes and connects all the
    sub-modules defined in my proposal:
    1. DRS: Directional Residual Stack for visual difference features.
    2. Text Encoder: To get a vector from the question.
    3. QDT: Question-Guided Tokenizer to select the most relevant difference tokens.
    4. MRM: Masked Residual Model for the self-supervised pre-training task.
    5. Head: The final Transformer Decoder that generates the answer.
    """

    def __init__(
        self,
        backbone,
        text_encoder,
        text_model_name,
        text_dim,
        text_proj_dim,
        text_finetune,
        topk,
        num_rows=3,
        num_cols=2,
        num_classes=1000, # This is the vocabulary size for the decoder
        max_ans_len=32
    ):
        super().__init__()

        # Vision Encoder (DRS+)
        self.drs = DirectionalResidualStack(backbone_name=backbone)

        # get number of output channels from DRS backbone, 512 for resnet-18
        # this will be C * 4 (512 * 4 = 2048)
        C = self.drs.out_channels
        c_all = C * 4  # [R+, R-, Rabs, signed]

        # Text Encoder
        # this module turn the input question (string) into a vector
        if text_encoder == "clinicalbert":
            # use pre-trained ClinicalBERT model
            self.text = ClinicalBERTText(
                model_name=text_model_name,
                d_txt=text_dim,
                proj_dim=text_proj_dim,
                fine_tune=text_finetune,
            )
            self.uses_hf = True
            q_dim = text_proj_dim
        else:
            # use simple hash-based text encoder as a baseline
            self.text = TinyText(d_txt=text_proj_dim)
            self.uses_hf = False
            q_dim = text_proj_dim

        # Question-guided Difference Tokenizer
        self.qdt = QuestionGuidedDifferenceTokenizer(
            c_img=c_all, # input channel dimension from DRS 
            d_txt=q_dim, # input dimension of the question vector
            k=topk, # number of tokens (K) to select
            num_rows=num_rows, # for zone-based pooling
            num_cols=num_cols # for zone-based pooling
        )

        # Masked Residual Model
        # initialize MRM for self-supervised pre-training task (Stage A)
        self.mrm = MaskedResidualModel(c_all=c_all, mask_ratio=0.6)
        
        # Answer Generation Head Decoder
        # The final modue that generates text anser
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
