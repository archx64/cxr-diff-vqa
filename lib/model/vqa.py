import torch
import torch.nn as nn
import logging
from .drs import DirectionalResidualStack
from .qdt import QuestionGuidedDifferenceTokenizer, TinyText, ClinicalBERTText
from .mrm import MaskedResidualModel
from .heads import TinyTransformerDecoder

logger = logging.getLogger(__name__)

class DiffVQAModel(nn.Module):
    """
    This is the main DRIFT-VQA model
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
        num_classes, # This is now the vocab_size
        max_ans_len,
        num_rows=3,
        num_cols=2,
    ):
        super().__init__()
        logger.info(f"Initializing DiffVQAModel (Decoder-Only) with backbone: {backbone}")

        # --- 1. Visual Difference Module (DRS) ---
        # this module takes two images and produces R+, R-, R_abs and signed featured maps
        self.drs = DirectionalResidualStack(backbone_name=backbone)
        C = self.drs.out_channels
        c_all = C * 4 # total channel dimension after concatenating all 4 difference maps
        logger.info(f"DRS output channels: {C}, Total visual dim: {c_all}")

        # --- 2. Text Encoder Module ---
        logger.info(f"Initializing text encoder: {text_encoder}")
        self.text = ClinicalBERTText(
            model_name=text_model_name,
            d_txt=text_dim,
            proj_dim=text_proj_dim,
            fine_tune=text_finetune,
        )
        q_dim = text_proj_dim # dimension of final question vector

        # --- 3. Question-Guided Tokenizer (QDT) ---
        # this module takes difference maps and the question vector
        # and performs cross-attention to select the top k-visual tokens
        self.qdt = QuestionGuidedDifferenceTokenizer(
            c_img=c_all,
            d_txt=q_dim,
            k=topk,
            num_rows=num_rows,
            num_cols=num_cols
        )
        
        # --- 4. Masked Residual Model (MRM) ---
        self.mrm = MaskedResidualModel(c_all=c_all, mask_ratio=0.4)

        # --- 5. Answer Generation Head (Decoder) ---
        # Hard-coded to the decoder, 'head' argument is removed.
        logger.info(f"Initializing Decoder Head. Vocab size: {num_classes}, Max len: {max_ans_len}")
        self.head = TinyTransformerDecoder(
            dim=c_all,
            vocab_size=num_classes,
            nlayer=3,
            nhead=8,
            max_len=max_ans_len
        )

    def forward(self, img_ref, img_cur, token_batch):
        """
        Main forward pass. Returns intermediate tensors for loss calculation.
        """

        # get visual maps from DRS
        # r is a dictionary: {'r_pos': ..., 'r_neg': ..., 'r_abs': ..., 'signed':...}
        r = self.drs(img_ref, img_cur)

        # get the question vector from text encoder
        q_vec = self.text(token_batch) 
        
        # 3. Use the QDT to select the most relevant visual tokens
        # sel_tokens: (Batch, k, c_all) - The k most important visual tokens
        # heatmap: (Batch, H, W) - The 2D attention map for visualization
        # gate_l1: A small loss for regularization
        sel_tokens, heatmap, gate_l1 = self.qdt(q_vec, r) 
        
        # run the MRM head (this is done in parallel, only its loss is used in training)
        feats_for_mrm = torch.cat(
            [r["r_pos"], r["r_neg"], r["r_abs"], r["signed"]], dim=1
        )
        mrm_out = self.mrm(feats_for_mrm)

        # Removed 'if classifier:' logic. Always return tensors for the decoder.
        return {
            "sel_tokens": sel_tokens,
            "heatmap": heatmap,
            "gate_l1": gate_l1,
            **mrm_out,
            **r,
        }