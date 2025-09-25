# Drift

The DRIFT-QA model is a lightweight, difference-aware medical Visual Question Answering (VQA) system designed to answer longitudinal, difference-focused questions about pairs of medical images, such as Chest X-rays. Its architecture is built around several novel components that address common challenges in longitudinal reasoning, such as inefficiency, direction blindness, hallucinations, and lack of trustworthiness.

## Architecture

### 1. Inputs

The model takes two primary inputs:

- Current Chest X-ray ($I_{cur}$). t
- Reference Chest X-ray ($I_{ref}$). 0
- A natural-language question ($Q$) that is difference-type.

### 2. Core Modules and Their Architecture

#### 2.1. Vision Backbone (Tiny & Frozen)

- Purpose: Extracts shallow features from the input images.
- Choices: A tiny, frozen backbone such as ResNet-18 or Tiny-ViT
• Output: Multi-scale feature maps, denoted as $F_{ref}^s, F_{cur}^s$ for different scales $s \in {1/8, 1/16}$, where each map $F_{\bullet}^s \in \mathbb{R}^{H_s \times W_s \times C}$ (e.g., C=192).

#### 2.2. Directional Residual Stack (DRS) / Directional Residual Composer

This module is designed for explicit directionality, distinguishing between worsening/new findings and improving/resolved findings.

- Computation of Three Maps: For each scale $s$, it computes three distinct residual maps from the shallow features:
  - $R^+ = $ ReLU($F_{cur}^s - F_{ref}^s$): Represents what emerged or worsened
  - $R^- = $ ReLU($F_{ref}^s - F_{cur}^s$): Represents what resolved or improved
  - $R| = |F_{cur}^s - F_{ref}^s|$: Represents the magnitude of change, which is direction-agnostic
- Stacking: These maps are stacked and concatenated to form $R^*$. A learned direction/type embedding $e_{dir} \in {+, -, |}$ is added. This approach differs from prior "single residual branch" methods by explicitly separating directions and magnitude.

#### 2.3. Text Encoder

- Purpose: Encodes the natural-language question (q).
- Choice: A frozen, small text encoder, such as a tiny transformer or BiGRU (e.g., 6M parameters).
- Output: Produces token embeddings $T \in \mathbb{R}^{L \times d}$ and a pooled summary vector $q̄ \in \mathbb{R}^{d}$.

#### 2.4. Question-Guided Difference Tokenizer (QDT)

The QDT acts as a minimal visual interface, aiming to select only a small number of relevant visual tokens

#### 2.5. Lightweight Answerer / Lightweight Fusion Head

This component is responsible for generating the model's answer.

- Options:
  - (A) Classifier (recommended): Fuses the K difference tokens with the pooled question summary $q̄$ using attention pooling ($z = \text{AttnPool}({\text{tokens}_K}, q̄)$). An MLP then produces logits over canonical classes such as {new, worsened, improved/resolved, unchanged} or {yes, no}.
  - (B) Tiny Generator (optional): For free-form text generation, it uses a small seq-to-seq LM (e.g., T5-small-like). LoRA adapters are added only to the cross-attention mechanism that consumes the K difference tokens, keeping the base LM frozen. This allows for open-ended phrasing, aligning with generative Med-VQA trends.

#### 2.6. Counterfactual Regularizer (CFA-Reg)

While primarily a training objective (Loss $L_{cf}$), it's a key architectural novelty for anti-hallucination.

- Mechanism: For each training instance $(I_{cur}, I_{ref}, Q, A)$, a counterfactual is created by swapping images ($I_0, I_t$) and/or negating change cues in the question (e.g., “new” $\leftrightarrow$ “resolved”, “worse” $\leftrightarrow$ “better”).
- Training: It uses a contrastive KL divergence between the model’s distributions on real vs. counterfactual inputs, pushing answers apart when evidence contradicts the change. This enforces directional consistency and reduces difference-hallucination.
- No-Change Detector Head: A light "no-change" detector head can be added from $R^*$ to calibrate abstention on ambiguous pairs.

#### 2.7. Self-Supervised Warmup (MRM: Masked Residual Modeling)

This is a pre-training strategy that allows the vision side of the model to become difference-sensitive without relying on radiology reports or external corpora.

- Process: The DRS and QDT modules are pre-trained by masking random patches in $R^*$ and then reconstructing them from the unmasked context.

#### 2.8. Evidence Projector

- Purpose: Generates faithful and low-cost visual evidence to help clinicians trust the model's answers.
- Mechanism: It back-projects the Top-K selected patch coordinates onto the image grid. These regions can optionally be blurred with a small Gaussian. The type color (+, -, |) is overlaid to visualize the direction of evidence, such as highlighting the left apical region where lung markings disappeared for a new pneumothorax. This provides a clear explanation for the model's output.

#### Key Design Principles and Contributions

- Efficiency (RQ1): The QDT uses a small set of K difference tokens (e.g., 12 vs thousands of full-image tokens), proving that most of the image is irrelevant and saving compute.
- Explicit Bi-directional Residuals (RQ2): The DRS explicitly separates "got worse/new" ($R^+$) from "got better/resolved" ($R^-$) signals, which is not explicitly factored in prior residual designs. This improves reasoning about clinical phrases like "worse," "better," "new," or "resolved".
- Counterfactual Regularizer (RQ3): This data-local regularizer targets hallucination control by training with counterfactual examples (swapping images, negating questions), reducing the tendency to claim changes that don't exist.
- Trustworthy Explanations (RQ4): The model provides faithful, low-cost visual evidence through token-selection maps (evidence heatmaps) directly derived from the QDT, without needing auxiliary supervision or retrieval databases.
This architecture aims to provide a compute-efficient and interpretable solution for longitudinal medical VQA, addressing the core problems of existing AI models in this domain
