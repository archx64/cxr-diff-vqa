# cxr-diff-vq

## New Architecture

- Inputs
  - $I_{cur}$ = current CXR
  - $I_{ref}$ = reference CXR
  - $q$ = natural-language question (difference-type)

- Directional Residual Stack (DRS) - explicit directionality
  - Extract shallow features with a tiny frozen backbone (e.g., ResNet-18/ViTtiny)
  - Compute three maps:
    - $R^{+} = f(I_{cur}) − f(I_{ref})$ (what emerged/worsened)
    - $R^{-} = f(I_{ref}) − f(I_{cur})$ (what resolved/improved)
    - $R∣ = |f(I_{cur}) − f(I_{ref})|$ (magnitude, direction-agnostic)

    - Concatenate $R*$. This is different from prior “single residual branch” by explicitly separating directions and magnitude.

- Question-Guided Difference Tokenizer (QDT) — minimal visual interface
  - Encode $q$ (frozen small text encoder).
  - Cross-attend $q$ over $R∗$ to produce K learned “difference tokens” (K ≪ fullpatch tokens).
  - Use a Top-k (Gumbel) selector to make tokens sparse and interpretable.
  - Export the selection map as an evidence heatmap at no extra cost.

- Lightweight Answerer — tiny language head with LoRA
  - Freeze a small seq-to-seq LM (e.g., T5-small-like) and add LoRA adapters only to the cross-attention that consumes the K difference tokens.
  - Generate free-form answers (handles open-ended phrasing better than fixed classification sets, in line with generative Med-VQA trends).

- Counterfactual Regularizer (CFA-Reg) — anti-hallucination
  - For each ($Iₜ, I₀, q, a$): create a counterfactual by swapping images and/or
  negating change cues in q (e.g., “new”↔“resolved”).
  - Train a contrastive KL between the model’s distributions on (real vs
  counterfactual), pushing answers apart when evidence contradicts the change.
  - Add a light “no-change” detector head from R∗ to calibrate abstention on
  ambiguous pairs.

- Self-Supervised Warmup (MRM: Masked Residual Modeling) — no reports
needed
  - Pre-train the DRS+QDT by masking random patches in R∗ and reconstructing
  them from unmasked context.
  - This teaches the vision side to be difference-sensitive without relying on
  radiology reports or external corpora (unlike PLURAL).

- New Research Questions
  - RQ1: Can longitudinal, difference focused questions be accurately answered using only a handful of question-guided difference tokens, instead of processing all image patches?
    - Input images: Chest X-rays taken two months apart.
    - Question: “Is there a new opacity in the right lower lobe?”
    - Dense baseline: The model processes thousands of visual tokens from both images.
    - Our idea: The model selects only ~8–16 difference tokens focused on the right lower
    lung region, guided by the question.
    - Why this matters: If the model answers correctly using just a few tokens, it proves
    that most of the image is irrelevant, and we can save compute while improving
    interpretability.

  - RQ2: Does explicitly modeling the direction of change between two images help the model distinguish clinical phrases like “new,” “worse,” “improved,” “resolved”?
    - Input images: Chest X-rays from Day 1 and Day 10.
    - Question: “Has the pleural effusion worsened?”
    - Direction-aware model: Compares Day 10 → Day 1 to capture whether effusion
    volume increased.
    - Direction-agnostic baseline: Uses only |difference| (absolute residual), which cannot
    distinguish “worse” vs “improved.”
    - Expected outcome: The directional model answers “Yes, effusion worsened”, while
    the residual-only model might just say “Change detected” without knowing in which
    direction.

  - RQ3:Can a counterfactual training signal such asswapping images or negating terms in the question reduce the model’s tendency to hallucinate differences?
    - Original:
      - Input images: Day 1 and Day 5 chest X-rays (no new findings).
      - Question: “Is there a new nodule?”
      - Naïve model output: “Yes, new nodule in right lung” (hallucination).
    - Counterfactual training: Swap image order + negate term:
      - Question: “Has the nodule resolved?”
      - Ground truth: “No.”
    - Impact: The model learns that if swapping/negating does not align with image
      evidence, the answer should be “No.”
    - Why this matters: This discourages false positives (hallucinating change when none
      exists).





  

## Old Architecture

### Regi-Fusion VQA Model Architecture & Training

- Two images:
  - Image A (Previous X-ray)
  - Image B (Current X-ray)

- **_Inputs to the model_**
  - Image A' (Registered Previous X-ray)
  - Image B (Current X-ray)
  - Deformation Field
  - Question (Text String)
  - Ground Truth Answer (Text String for training)

- **_Feature Encoders_**
  - Model: A pre-trained Vision Transformer (ViT-B/16).
  - Inputs: Image B, Image A', and the Deformation Field. The deformation field's 2 channels can be repeated to a 3rd channel to fit the ViT's input requirements.
  - Process: The encoder is shared. Each of the three inputs is passed through the ViT independently to extract high-level features.
  - Output: Three feature vectors: features_B, features_A', features_D. The output dimension of a base ViT is typically 768, so each vector has a shape of [batch_size, 768].- Training Status: FROZEN initially. In the first stage of training, you will not update the weights of the ViT. This is crucial for stability. After the rest of your model has started to learn, you can unfreeze the ViT and fine-tune it with a very low learning rate
  
- **_Text Encoder_**
  - Model: A pre-trained ClinicalBERT from a library like Hugging Face.
  - Inputs: Question and Ground Truth Answer.
  - Process: The encoder tokenizes the text and passes it through the BERT model.
  - Output: Two feature vectors: features_Q and features_Ans, each with a shape of[batch_size, 768].
  - Training Status: FROZEN initially, then optionally fine-tuned along with the ViT.

- **_Difference Fusion Module_ (The Brain)**
  - Model: A small Transformer Encoder Layer. This is a standard module with self-attention and a feed-forward network.
  - Input: The three image feature vectors (features_B, features_A', features_D).
  - Process:
        1. First, calculate a raw difference: raw_diff = features_B - features_A'.
        2. Concatenate the three vectors into a sequence: [raw_diff, features_B, features_D].
        3. Pass this sequence through the Transformer Encoder Layer. This allows the model to use attention to weigh the importance of each feature type and learn their complex interactions.
  - Output: A single, refined vector that represents the change: fused_diff_features (Shape: [batch_size, 768]).
  - Training Status: TRAINED FROM SCRATCH. This module's weights are initialized randomly and learn during training.'

- **_Contrasive Head_**
  - Model: A simple Multi-Layer Perceptron (MLP). Two linear layers with a ReLU activation in between.
  - Input: The fused_diff_features from the fusion module and the features_Q from the text encoder.
  - Process:
    - The two input vectors are added or concatenated.
    - This combined vector is passed through the MLP to project it into the contrastive learning space.
  - Output: A query_vector (e.g., shape [batch_size, 256]) used for calculating the contrastive loss against the features_Ans (which would also be projected by a similar MLP).
  - Training Status: TRAINED FROM SCRATCH.

- **_Generation Head_**
  - Model: A standard Transformer Decoder.
  - Input: The fused_diff_features is used as the memory (the context) for the decoder's cross-attention layers. The input to the decoder itself is the sequence of previously generated answer tokens, starting with a [START] token.
  - Process: At each step, the decoder attends to the fused difference features and the question to predict the next word in the answer.
  - Output: The final Predicted Answer (text string).
  - Training Status: TRAINED FROM SCRATCH.

- Final Loss Function
  - The total loss function that you backpropagate through the trainable parts of the model is a weighted sum:
    - $Total Loss = L_{generation} + α * L_{contrastive} $
    - $L_{generation}$: Standard Cross-Entropy loss between the Predicted Answer and the Ground
    Truth Answer.
    - $L_{contrastive}$: An InfoNCE loss that tries to pull the query_vector and the correct Answer vector
    together while pushing it away from all other answers in the batch.
    - α: A hyperparameter (e.g., 0.1) that you tune to balance the two learning objectives.
