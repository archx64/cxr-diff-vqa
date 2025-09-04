# Questions

## Q1

We're going to focus only on "difference" questions. How will the quality of the dataset and model be impacted if the other types of questions are removed from the dataset?

## Answer to Q1

Focusing only on "difference" questions is a valid and common approach that will sharpen your model's specialization, but it comes with a trade-off in data volume and potentially lost learning opportunities.

### Impact on Dataset Quality

Filtering for only "difference" questions is a standard and often necessary step for this specific research. Several of the papers you've provided do the exact same thing.

#### Pro (Increased Specificity)

The quality of your dataset for the core task improves. You are left with a clean, highly specific benchmark that contains only examples directly relevant to your thesis. The VED model paper, for instance, explicitly states, "For the purposes of this work, only the 'difference' question type is considered, as it aligns with the current state-of-the-art focus in the field". The ReAl paper does the same, concentrating exclusively on the "difference" subset.

#### Con (Reduced Volume)

The main drawback is the significant reduction in data. The MIMIC-Diff-VQA dataset contains 700,703 total question-answer pairs, but only about 164,000 (or 23%) are "difference" questions. You will be training on a much smaller dataset, which can sometimes make it harder for a model to generalize.

### Impact on Model Quality & Performance

#### Pro (Specialization)

Your model will become a highly specialized expert at answering only difference questions. By not having to learn the patterns for six other question types (like "location" or "presence"), its entire capacity can be dedicated to mastering the complex reasoning required for longitudinal comparison.

#### Con (Loss of Contextual Learning)

You may lose out on some "cross-task" learning. For example, learning to answer a "location" question ("Where is the opacity?") or an "abnormality" question ("What abnormality is seen?") could provide the model with a stronger foundational understanding of radiological concepts. This broader knowledge might indirectly help it answer a "difference" question more accurately. Models like PLURAL and RegioMix were evaluated on all question types, suggesting there is value in training on the full dataset to create a more versatile and robust model.
___

## Q2

What does "negating the question" mean? Explain with an example.

## Answer to Q2

"Negating the question" means inverting the clinical premise of the question by swapping key "change" words with their logical opposites.

The question is not grammatically negative (e.g., by adding "not"). Instead, a logical counterfactual is created to ask about the opposite clinical outcome.

### Example

Let's break down an example from your research questions:

__Original Question:__ "Has the pleural effusion worsened?"

The "change cue" here is worsened. The question is probing for an increase or progression of a finding.

Find the Logical Opposite: The clinical opposite of "worsened" is "improved" or "resolved."

Construct the Negated Question: You swap the change cue with its opposite.

__Negated Question:__ "Has the pleural effusion improved?"

### Why this is done

The goal of the __Counterfactual Regularizer (CFA-Reg)__ is to create a "logical trap" for the model. By showing the model the same set of images but asking it both the original and the negated question, The model learns to be sensitive to the precise wording and to ground its answers in the visual evidence.

### Swapping Images

Swapping the images serves the same fundamental purpose as negating the question. Both actions are designed to create a counterfactual pair to the original training example. The goal is to create a new, logically opposite scenario to test the model's consistency and robustness against hallucinations.

Method 1: Negating the Question

- __Images:__ [Past Image, Current Image] (Evidence shows worsening)
- __Question:__ "Has it improved?"
- __Result:__ The question's premise contradicts the visual evidence.
- __Expected Answer:__ "No"

Method 2: Swapping the Images

- __Images:__ [Current Image, Past Image] (Evidence now shows improvement)
- __Question:__ "Has it worsened?"
- __Result:__ The visual evidence now contradicts the question's premise.
- __Expected Answer:__ "No"

___

## Q3

Explain how R_{abs}, R_{-} and R{+} are calculated? Why are they calculated? Why are the concatenated at the end?

## Answer to Q3

### How residual maps are calculated

The three residual maps are calculated from the shallow features $(f)$ extracted by the tiny, frozen backbone from the current $(I_{cur})$ and reference $(I_{ref})$ images.

- $R_+ = f(I_{cur}) - f(I_{ref})$
  - This is a simple subtraction. It calculates what is present in the current image but was absent or less prominent in the reference image.

- $R_- = f(I_{ref}) - f(I_{cur})$
  - This is the reverse subtraction. It calculates what was present in the reference image but is now absent or less prominent in the current image.

- $R_{abs} = |f(I_{cur}) - f(I_{ref})|$
  - This calculates the absolute difference between the features.

### Why they are concatenated

The three maps $(R_+, R_-, R_{abs})$ are concatenated at the end to create a single, multi-channel feature map (R).

This is done to provide the downstream model—the Question-Guided Tokenizer—with a rich and complete representation of the visual difference in a single, efficient tensor. By stacking them together, the downstream model efficiently learns:

"For every location in the image, here is the evidence for what has worsened (Channel 1), what has improved (Channel 2), and the total amount of change (Channel 3)."

This explicit, multi-faceted signal is much more informative than a single residual map and is designed to make it easier for the model to answer complex, direction-specific clinical questions.

___

## Q4

What is Question Guided Tokenizer? Why is evidence heat map and K-difference tokens required in Question Guided Tokenizer?

## Answer to Q4

### Question-Guided Difference Tokenizer

The Question-Guided Difference Tokenizer (QDT) is the core innovation of the architecture responsible for efficiency and interpretability. Its job is to distill the vast amount of visual information from the Directional Residual Stack (DRS) down to a small, manageable set of the most relevant features needed to answer a specific question.

Instead of forcing the language model to process thousands of visual features from the entire image (the "brute-force" approach), the QDT intelligently selects only the crucial pieces of evidence.

### K-Differnce Tokens

The K-Difference Tokens are the primary output of the QDT. They are a small, fixed number (e.g., K=16) of feature vectors that represent the most important visual evidence from the residual maps, as determined by the question.

- __Purpose:__ Efficiency. These K-tokens are the only visual information the Lightweight Answer Predictor ever sees. By compressing all the relevant visual data into a handful of tokens, you drastically reduce the computational load on the language model, making the entire architecture faster and more efficient.

- __Function:__ They serve as the bridge between the vision and language components of your model. The language model uses cross-attention to "look at" these K-tokens to find the visual evidence it needs to generate an answer.

### Evidence Heatmap

The Evidence Heatmap is a valuable byproduct of the QDT's selection process. It is a 2D visualization that highlights the regions in the original image that the model paid the most attention to when selecting its K-Difference Tokens.

- __Purpose:__ Interpretability. This heatmap is not required for the model to generate an answer, but it is crucial for you and for a potential clinical user. It provides a direct, visual explanation of the model's reasoning.

- __Function:__ It allows you to verify that the model is "looking" at the correct anatomical location when answering a question. For example, if the question is about the "right lower lobe," the evidence heatmap should show high activation in that specific area of the chest X-ray. This helps to build trust in the model and makes it easier to debug when it makes a mistake.

___

## Q5

What is contrastive KL loss in counterfactual regularizer?

## Answer to Q5

The Contrastive KL Loss is a training technique you designed in your architecture to make the model more robust and less likely to hallucinate.

### How it works

The process involves comparing the model's output for two different scenarios:

1. The Real Scenario: Your model processes a real image pair and a real question (e.g., "Has the nodule worsened?"). It doesn't just output a single word; it produces a probability distribution $(P_{real})$ over its entire vocabulary for the answer. For a correct answer, the probability for "Yes" would be high.

2. The Counterfactual Scenario: You create a logical opposite, for example, by swapping the images so the evidence now shows improvement. You feed this counterfactual input into the same model, which produces a second probability distribution $(P_{cf})$. In this case, the probability for "No" should be high.

The Contrastive KL Loss then measures the difference between these two probability distributions. The goal is to make them as different as possible. By minimizing this loss, you train the model to be highly sensitive to contradictions between the question and the visual evidence, thus reducing the chance it will "hallucinate" an answer that isn't strongly supported.

___

## Q6

Why are the models frozen in architecture?

## Answer to Q6

The models are frozen for two primary reasons: to achieve extreme computational efficiency and to ensure training stability.

Think of it as assembling a team of world-class specialists (the frozen backbones) and a small, agile project manager (the trainable parts). Specialists dont't need to be retrained but project manager needs to learn how to query them effectively for the specific task.

Yes, this is a crucial design choice. In your new architecture, the models are frozen for two primary reasons: to achieve extreme computational efficiency and to ensure training stability.

Think of it as assembling a team of world-class specialists (the frozen backbones) and a small, agile project manager (the trainable parts). You don't need to retrain the specialists; you just need to teach the project manager how to query them effectively for your specific task.

### To Maximize Efficiency ⚡

The model is intentionally designed to be lightweight, in stark contrast to the "brute-force" SOTA models. Freezing is the key to this efficiency.

- __Fewer Parameters to Train:__ The vast majority of the model's parameters—in the vision backbone, text encoder, and main language model—are locked. The only parts that are trained from scratch are the very small components like the QDT's cross-attention module and the LoRA adapters.

- __Benefits:__ This drastically reduces the GPU memory required and makes each training iteration much faster, as the computationally expensive backpropagation step is only performed on a tiny fraction of the total parameters.

### To Ensure Training Stability 🛡️

Freezing protects the powerful, pre-trained knowledge stored in the large backbones.

- __Preserving Knowledge:__ The frozen encoders act as reliable, high-quality feature extractors. Their deep understanding of visual and linguistic patterns is preserved.

- __Preventing Corruption:__ The new, trainable parts of your model start with random weights. If you were to train everything at once, the large, chaotic gradients from these new parts could corrupt and "forget" the valuable knowledge in the pre-trained models. Freezing the backbones allows the new, smaller modules to learn in a stable environment.

### To Enable Parameter-Efficient Fine-Tuning (PEFT)

Freezing the main language model and only training LoRA (Low-Rank Adaptation) adapters is a state-of-the-art technique. It allows you to adapt a massive language model to a new, specific task by training less than 1% of its total parameters. This is the ultimate expression of the efficiency-focused philosophy of your new architecture.

___

## Q7

What is Masked Residual Modelling? How does it work? Why is it required? What is reconstruction loss?

## Answer to Q7

a breakdown of Masked Residual Modeling

### Masked Residual Modeling

Masked Residual Modeling (MRM) is a self-supervised pre-training strategy you designed for your architecture. Its purpose is to teach the vision components of your model (the DRS and QDT backbone) how to understand the structure and context of radiological differences before they are ever trained on the actual question-answering task.

The key advantage is that it is self-supervised, meaning it learns without needing any text data, radiology reports, or human-provided labels.

### How it works

The process is directly inspired by the masked modeling techniques used in famous models like BERT and Vision Transformers (MAE):

- __Generate Residuals:__ First, an input image pair is passed through the Directional Residual Stack (DRS) to create the three difference maps $(R_+, R_−, R_{abs})$.

- __Mask Random Patches:__ A significant portion of these residual maps is randomly hidden or "masked."

- __Predict the Missing Patches:__ The model's vision backbone (the QDT backbone, in this case) is then given the corrupted, masked maps and tasked with predicting and reconstructing the missing patches based solely on the surrounding, unmasked context.

### Why It Is Required

MRM is required to pre-train your vision system to be difference-sensitive.

By forcing the model to reconstruct hidden parts of the difference maps, it learns powerful, contextual representations about what constitutes a meaningful radiological change. For example, it learns the typical shape and texture of a "worsening opacity" or a "resolving effusion."

This "warm-up" makes the model much more effective and data-efficient when you later fine-tune it on the main VQA task, as it already has a strong foundational understanding of the visual difference signals. This is what allows your model to be self-contained, avoiding the need for external radiology reports for pre-training, unlike models like PLURAL.

## What is Reconstruction Loss?

Reconstruction Loss is the metric used to guide the MRM training process. It measures the difference between the patches the model predicted and the original, unmasked patches it was trying to guess.

Typically, this is a simple mathematical function like Mean Squared Error (MSE) or L1 Loss, which calculates the average pixel-wise difference between the ground truth and the prediction. The model's goal during this pre-training stage is to minimize this loss, which, in turn, makes its reconstructions more accurate and its internal representations more robust.
