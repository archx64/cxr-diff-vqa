import torch
import torch.nn as nn
import logging

from .drs import DirectionalResidualStack
from .qdt import QuestionGuidedDifferenceTokenizer, ClinicalBERTText
from .mrm import MaskedResidualModel
from .heads import TinyTransformerDecoder

logger = logging.getLogger(__name__)

class DiffVQAModel(nn.Module):
    """
    The main DRIFT-VQA model.
    
    Includes support for ablation studies via the 'ablation_no_direction' flag.
    If True, the model uses only the absolute difference (magnitude) and ignores 
    directional information (new vs. gone), testing the hypothesis that directionality matters.
    """
    def __init__(
        self,
        backbone="resnet18",
        text_encoder="clinicalbert",
        text_model_name="emilyalsentzer/Bio_ClinicalBERT",
        text_dim=768,
        text_proj_dim=256,
        text_finetune=False,
        topk=64,
        num_rows=3,
        num_cols=2,
        num_classes=8000,
        max_ans_len=48,
        mask_ratio=0.5,
        ablation_no_direction=False 
    ):
        super().__init__()
        self.ablation_no_direction = ablation_no_direction
        
        mode_str = "ABLATION (Only R_abs)" if ablation_no_direction else "FULL (R+, R-, R_abs, Signed)"
        logger.info(f"Initializing DiffVQAModel in {mode_str} mode with backbone: {backbone}")

        # --- 1. Vision Encoder (DRS) ---
        self.drs = DirectionalResidualStack(backbone_name=backbone)
        C = self.drs.out_channels
        
        # --- ABLATION LOGIC ---
        # Determine the channel dimension fed to QDT/Decoder/MRM
        if self.ablation_no_direction:
            # Only 1 map used: R_abs
            c_all = C 
        else:
            # 4 maps used: R+, R-, R_abs, Signed
            c_all = C * 4  
        # ----------------------

        # --- 2. Text Encoder ---
        self.text = ClinicalBERTText(
            model_name=text_model_name,
            d_txt=text_dim,
            proj_dim=text_proj_dim,
            fine_tune=text_finetune,
        )
        q_dim = text_proj_dim

        # --- 3. Question-Guided Tokenizer (QDT) ---
        # c_img will be smaller in ablation mode (C instead of 4*C), saving compute
        self.qdt = QuestionGuidedDifferenceTokenizer(
            c_img=c_all, d_txt=q_dim, k=topk, num_rows=num_rows, num_cols=num_cols
        )

        # --- 4. Masked Residual Model (MRM) Head ---
        # Used for pre-training (Stage A). Needs to match the feature size.
        self.mrm = MaskedResidualModel(c_all=c_all, mask_ratio=mask_ratio)

        # --- 5. Decoder Head ---
        logger.info(f"Initializing decoder head with input dim {c_all}.")
        self.head = TinyTransformerDecoder(
            dim=c_all,
            vocab_size=num_classes,
            nlayer=3,
            nhead=8,
            max_len=max_ans_len,
        )

    def forward(self, img_ref, img_cur, token_batch):
        """
        Main forward pass.
        Calculates differences -> Selects Tokens -> Returns intermediate features.
        """
        # 1. Get raw difference dictionary from DRS
        # Returns: {'r_pos':..., 'r_neg':..., 'r_abs':..., 'signed':...}
        r = self.drs(img_ref, img_cur)

        # 2. Get text vector
        q_vec = self.text(token_batch)

        # --- 3. PREPARE FEATURES FOR ABLATION ---
        if self.ablation_no_direction:
            # ABLATION: Use ONLY the absolute difference magnitude
            # The model loses the ability to distinguish "new" vs "gone" features
            visual_feats = r["r_abs"]
        else:
            # STANDARD: Concatenate ALL directional information
            visual_feats = torch.cat(
                [r["r_pos"], r["r_neg"], r["r_abs"], r["signed"]], dim=1
            )
        # ----------------------------------------

        # 4. Pass prepared features to QDT
        # (Relies on the updated QDT forward method that accepts a generic tensor)
        sel_tokens, heatmap, gate_l1 = self.qdt(q_vec, visual_feats)

        # 5. Pass prepared features to MRM
        mrm_out = self.mrm(visual_feats)

        return {
            "sel_tokens": sel_tokens,
            "heatmap": heatmap,
            "gate_l1": gate_l1,
            **mrm_out,
            **r,
        }