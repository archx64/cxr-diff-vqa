# Protoype

Here is the code for DRIFT VQA including the sample of the dataset files.

## Dataset

### Medical-Diff-VQA

There are one JSON file and two CSV files in Medical Diff VQA dataset

#### all_diseases.json

The KeyInfo intermediate dataset, which comprises key information such as abnormalities, location, type, and level needed for the extraction of the final VQA difference dataset. For "view" questions, this information will be directly retrieved from the mimic_all.csv. For "difference" questions, answers are acquired by comparing the KeyInfo of two studies (visits). The columns include the following:

study_id: study id in MIMIC-CXR
subject_id: subject id in MIMIC-CXR
entity: The identified disease name.
Disease name: a string represents the disease name
entity_name: same as the disease name
Location: The word indicating the location of the disease.
Type: The word describing the type or category of the disease.
Level: The word indicating the level or severity of the disease.
post_location: The word indicating the location of the disease that appears after the disease name.
Location2: Same as "Location". Provided as a backup in case there are multiple locations.
Type2: Same as "Type". Provided as a backup in case there are multiple types or categories.
Level2: Same as "Level". Provided as a backup in case there are multiple levels or severities.
post_location2: Same as "post_location". Provided as a backup in case there are multiple location words that appear after the disease name.
no_entity: a list containing the names of all diseases that are identified as not existing.

A sample of the JSON file looks like this

```json
[
    {
        "study_id": "55088298",
        "subject_id": "18936629",
        "entity": {
            "pneumothorax": {
                "entity_name": "pneumothorax",
                "location": [
                    "left"
                ],
                "type": null,
                "level": [
                    "minimal"
                ],
                "post_location": null,
                "location2": null,
                "type2": null,
                "level2": null,
                "post_location2": null
            },
            "atelectasis": {
                "entity_name": "atelectasis",
                "location": [
                    "left",
                    "basal"
                ],
                "type": null,
                "level": [
                    "mild"
                ],
                "post_location": "the soft tissues",
                "location2": null,
                "type2": null,
                "level2": null,
                "post_location2": null
            }
        },
        "no_entity": [
            "edema"
        ]
    }
]
```

#### mimic_pair_questions.csv

This is the final generated question-answer pairs of the difference VQA dataset. The columns include the following:

study_id: main study_id in MIMIC-CXR.
subject_id: subject id in MIMIC-CXR.
ref_id: reference study_id in MIMIC-CXR.
question_type: abnormality/location/level/view/type/presence/difference.
question
answer
split: train/val/test split.

10 rows of the csv file looks like this

|    |   study_id |   subject_id |   ref_id | question_type   | question                                                   | answer                                                                                                                                                                                                               | split   |
|---:|-----------:|-------------:|---------:|:----------------|:-----------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------|
|  0 |   55088298 |     18936629 | 54989200 | abnormality     | what abnormalities are seen in the soft tissues?           | atelectasis                                                                                                                                                                                                          | train   |
|  1 |   55088298 |     18936629 | 54989200 | presence        | is there edema?                                            | no                                                                                                                                                                                                                   | train   |
|  2 |   55088298 |     18936629 | 54989200 | location        | is the atelectasis located on the left side or right side? | left side                                                                                                                                                                                                            | train   |
|  3 |   55088298 |     18936629 | 54989200 | level           | what level is the pneumothorax?                            | minimal                                                                                                                                                                                                              | train   |
|  4 |   55088298 |     18936629 | 54989200 | difference      | what has changed compared to the reference image?          | the main image has an additional finding of pneumothorax than the reference image. the main image is missing the findings of fracture, lung opacity, and pleural effusion than the reference image.                  | train   |
|  5 |   51480826 |     18936629 | 50632107 | abnormality     | what abnormalities are seen in this image?                 | pleural effusion, pneumothorax, atelectasis, lung opacity                                                                                                                                                            | train   |
|  6 |   51480826 |     18936629 | 50632107 | presence        | is there pneumothorax in the right apical area?            | no                                                                                                                                                                                                                   | train   |
|  7 |   51480826 |     18936629 | 50632107 | location        | where is the pneumothorax?                                 | left apical area                                                                                                                                                                                                     | train   |
|  8 |   51480826 |     18936629 | 50632107 | level           | what level is the pleural effusion?                        | small                                                                                                                                                                                                                | train   |
|  9 |   50632107 |     18936629 | 51480826 | difference      | what has changed compared to the reference image?          | the main image has additional findings of consolidation, and pneumonia than the reference image. the main image is missing the findings of pleural effusion, pneumothorax, and atelectasis than the reference image. | train   |

#### mimic_all.csv

This file includes the metadata from MIMIC-CXR and labels from MIMIC-CXR-JPG. The labels are for reference purposes only. The columns are:

subject_id: subject id in MIMIC-CXR
study_id: study_id in MIMIC-CXR
labels: This section contains the labels that were extracted from MIMIC-CXR-JPG, but they are only for reference purposes.
Atelectasis
Cardiomegaly
Consolidation
Edema
Enlarged Cardiomediastinum
Fracture
Lung Lesion
Lung Opacity
Pleural Effusion
Pneumonia
Pneumothorax
Pleural Other
Support Devices
No Finding
dicom_id: dicom_id in MIMIC-CXR
view: "PA" or "AP" view of the image
split: train/val/test split used in MIMIC-CXR-JPG. It is only for reference purposes.
study_date: StudyDate in MIMIC-CXR-JPG
study_order: an integer that indicates the order number of a patient's entire visit history. Each patient may have multiple visits, but we select only two visits for pair comparison in order to create the final difference VQA dataset.

10 rows of the csv file looks like this

|    |   subject_id |   study_id |   Atelectasis |   Cardiomegaly |   Consolidation |   Edema |   Enlarged Cardiomediastinum |   Fracture |   Lung Lesion |   Lung Opacity |   No Finding |   Pleural Effusion |   Pleural Other |   Pneumonia |   Pneumothorax |   Support Devices | dicom_id                                     | view             | split   |   study_date |   study_order |
|---:|-------------:|-----------:|--------------:|---------------:|----------------:|--------:|-----------------------------:|-----------:|--------------:|---------------:|-------------:|-------------------:|----------------:|------------:|---------------:|------------------:|:---------------------------------------------|:-----------------|:--------|-------------:|--------------:|
|  0 |     10000032 |   50414267 |           nan |            nan |             nan |     nan |                          nan |        nan |           nan |            nan |            1 |                nan |             nan |         nan |            nan |               nan | 02aa804e-bde0afdd-112c0b34-7bc16630-4e384014 | postero-anterior | train   |     21800506 |             1 |
|  1 |     10000032 |   53189527 |           nan |            nan |             nan |     nan |                          nan |        nan |           nan |            nan |            1 |                nan |             nan |         nan |            nan |               nan | 2a2277a9-b0ded155-c0de8eb9-c124d10e-82c5caab | postero-anterior | train   |     21800626 |             2 |
|  2 |     10000032 |   53911762 |           nan |            nan |             nan |     nan |                          nan |        nan |           nan |            nan |            1 |                nan |             nan |         nan |            nan |               nan | 68b5c4b1-227d0485-9cc38c3f-7b84ab51-4b472714 | antero-posterior | train   |     21800723 |             3 |
|  3 |     10000032 |   56699142 |           nan |            nan |             nan |     nan |                          nan |        nan |           nan |            nan |            1 |                nan |             nan |         nan |            nan |               nan | ea030e7a-2e3b1346-bc518786-7a8fd698-f673b44c | antero-posterior | train   |     21800805 |             4 |
|  4 |     10000764 |   57375967 |           nan |            nan |               1 |     nan |                          nan |        nan |           nan |            nan |          nan |                nan |             nan |          -1 |            nan |               nan | 096052b7-d256dc40-453a102b-fa7d01c6-1b22c6b4 | antero-posterior | train   |     21321015 |             1 |
|  5 |     10000898 |   50771383 |           nan |            nan |             nan |     nan |                          nan |        nan |           nan |            nan |            1 |                nan |             nan |         nan |            nan |               nan | 2a280266-c8bae121-54d75383-cac046f4-ca37aa16 | postero-anterior | train   |     21880312 |             2 |
|  6 |     10000898 |   54205396 |           nan |            nan |             nan |     nan |                          nan |        nan |           nan |            nan |            1 |                nan |             nan |         nan |            nan |               nan | b75df1bd-0f22d631-52d73526-2ae7b85a-d843b39d | postero-anterior | train   |     21880113 |             1 |
|  7 |     10000935 |   50578979 |           nan |            nan |             nan |      -1 |                          nan |        nan |           nan |             -1 |          nan |                  1 |             nan |           1 |            nan |               nan | d0b71acc-b5a62046-bbb5f6b8-7b173b85-65cdf738 | antero-posterior | train   |     21871016 |             6 |
|  8 |     10000935 |   51178377 |           nan |            nan |             nan |     nan |                          nan |        nan |           nan |              1 |          nan |                nan |             nan |          -1 |            nan |               nan | 9b314ad7-fbcb0422-6db62dfc-732858d0-a5527d8b | antero-posterior | train   |     21870823 |             4 |
|  9 |     10000935 |   55697293 |           nan |            nan |             nan |     nan |                          nan |        nan |           nan |            nan |            1 |                nan |             nan |         nan |            nan |               nan | c50494f1-90e2bff5-e9189550-1a4562fd-6ab5204c | postero-anterior | train   |     21870226 |             2 |

### MIMIC-CXR-JPG

Images are provided in individual folders. An example of the folder structure for a single patient's images is as follows:

```bash
files/
  p10/
    p10000032/
      s50414267/
        02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
        174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962.jpg
      s53189527/
        2a2277a9-b0ded155-c0de8eb9-c124d10e-82c5caab.jpg
        e084de3b-be89b11e-20fe3f9f-9c8d8dfe-4cfd202c.jpg
      s53911762/
        68b5c4b1-227d0485-9cc38c3f-7b84ab51-4b472714.jpg
        fffabebf-74fd3a1f-673b6b41-96ec0ac9-2ab69818.jpg
      s56699142/
        ea030e7a-2e3b1346-bc518786-7a8fd698-f673b44c.jpg
```

## File Layout

```bash
lib/
    dataset.py
    models/
        __init__.py
        drs.py
        qdt.py
        mrm.py
    losses.py
    utils.py
    heads.py
    negate.py
    phrases.py
    utils.py
src/
    train.py
```

___

## Directional Residual Stack

`models/drs.py`

```python
# models/drs.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DirectionalResidualStack(nn.Module):
    """
    Frozen backbone -> 1x1 conv + GroupNorm calibration -> directional residuals.
    Returns R+ (cur-ref, relu), R- (ref-cur, relu), R_abs, and signed residual.
    """
    def __init__(self, backbone_name="resnet50", out_index=-1, freeze_backbone=True):
        super().__init__()
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
            nn.Conv2d(ch*3, ch, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(ch, ch, 1, bias=False)
        )

        self.out_channels = ch

    @torch.no_grad()
    def encode(self, x):
        return self.backbone(x)[0]  # (B,C,H,W)

    def forward(self, img_ref, img_cur):
        f_ref = self.calib(self.encode(img_ref))
        f_cur = self.calib(self.encode(img_cur))
        signed = f_cur - f_ref
        r_pos  = F.relu(signed)
        r_neg  = F.relu(-signed)
        r_abs  = signed.abs()
        return {"r_pos": r_pos, "r_neg": r_neg, "r_abs": r_abs, "signed": signed}

    def alignment_loss(self, r_pos, r_neg, r_abs, signed):
        combo = torch.cat([r_pos, r_neg, r_abs], dim=1)
        pred  = self.align(combo)
        return F.mse_loss(pred, signed)

```

___

## Question-Guided Tokenizer (QDT+) with zones & gated top-k

`models/qts.py`

```python
# models/qdt_plus.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuestionGuidedDifferenceTokenizer(nn.Module):
    """
    Concatenate residual maps (R+, R-, Rabs, optional signed conv) -> tokens.
    Add zone tokens (pooled by precomputed masks).
    Cross-attend tiny text embedding to visual tokens, produce gated top-k and heatmap.
    """

    def __init__(
        self, c_img, d_txt=256, k=64, num_rows=3, num_cols=2, use_zone_bias=True
    ):
        super().__init__()
        self.k = k
        self.use_zone_bias = use_zone_bias

        self.txt_to_img = nn.Linear(d_txt, c_img)  # project q to img-dim
        self.tok_proj = nn.Linear(c_img, c_img)  # residual token projection
        self.scale = c_img**-0.5

        # tiny gate over attention scores (per token)
        self.gate = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

        # Zone adjacency bias (fixed grid adjacency)
        self.num_rows, self.num_cols = num_rows, num_cols
        Z = num_rows * num_cols
        adj = torch.zeros(Z, Z)
        for r in range(num_rows):
            for c in range(num_cols):
                i = r * num_cols + c
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < num_rows and 0 <= cc < num_cols:
                        j = rr * num_cols + cc
                        adj[i, j] = 1.0
        self.register_buffer("zone_adj", adj)

        self.cached_masks = None  # filled on first forward

    def forward(self, q_vec, r_maps):
        """
        q_vec: (B, d_txt)
        r_maps: dict with keys r_pos, r_neg, r_abs, signed (each B,C,H,W). 'signed' optional.
        """
        r_pos, r_neg, r_abs = r_maps["r_pos"], r_maps["r_neg"], r_maps["r_abs"]
        signed = r_maps.get("signed", None)

        B, C, H, W = r_pos.shape
        if self.cached_masks is None:
            self.cached_masks = self.make_zone_masks(
                H, W, self.num_rows, self.num_cols
            ).to(
                r_pos.device
            )  # (Z,H,W)

        feats = torch.cat([r_pos, r_neg, r_abs], dim=1)  # (B,3C,H,W)
        if signed is not None:
            feats = torch.cat([feats, signed], dim=1)  # (B,4C,H,W)
        C_all = feats.shape[1]

        tokens = feats.flatten(2).transpose(1, 2)  # (B,HW,C_all)
        tokens = self.tok_proj(tokens)  # (B,HW,C_all)

        # zone tokens: pool per zone mask
        Z = self.cached_masks.shape[0]
        zone_tokens = []
        for z in range(Z):
            m = self.cached_masks[z]  # (H,W)
            m = m.view(1, 1, H, W)
            pooled = (feats * m).sum(dim=(2, 3)) / (m.sum() + 1e-6)  # (B,C_all)
            zone_tokens.append(pooled)
        zone_tokens = torch.stack(zone_tokens, dim=1)  # (B,Z,C_all)

        # concatenate tokens + zone tokens
        mem = torch.cat([tokens, zone_tokens], dim=1)  # (B,HW+Z,C_all)

        # attention from question to memory
        q = self.txt_to_img(q_vec).unsqueeze(1)  # (B,1,C_all)
        att = (q @ mem.transpose(1, 2)) * self.scale  # (B,1,HW+Z)
        att = att.transpose(1, 2)  # (B,HW+Z,1)
        gated = self.gate(att) * att  # gate is between 0 and 1 elementwise
        att_scores = gated.squeeze(-1)  # (B,HW+Z)
        att_weights = F.softmax(att_scores, dim=-1)  # (B,HW+Z)

        # evidence heatmap over spatial tokens only
        heat_spatial = att_weights[:, : H * W].reshape(B, H, W)

        # top-k selection over all tokens (spatial + zones)
        k = min(self.k, mem.shape[1])
        topk_vals, topk_idx = att_weights.topk(k, dim=-1)
        batch = torch.arange(B, device=mem.device)[:, None]
        sel_tokens = mem[batch, topk_idx]  # (B,k,C_all)

        # small sparsity regularizer (encourage the gate to actually reject)
        gate_l1 = gated.abs().mean()

        return sel_tokens, heat_spatial, gate_l1

    def make_zone_masks(self, h, w, num_rows=3, num_cols=2):
        """
        Simple grid-based lung zones: (num_rows x num_cols).
        Returns a tensor (Z, H, W) with binary masks.
        """
        masks = []
        row_edges = torch.linspace(0, h, steps=num_rows + 1).long()
        col_edges = torch.linspace(0, w, steps=num_cols + 1).long()
        for i in range(num_rows):
            for j in range(num_cols):
                m = torch.zeros(h, w, dtype=torch.float32)
                m[row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]] = (
                    1.0
                )
                masks.append(m)
        return torch.stack(masks, dim=0)  # (Z,H,W)


class TinyText(nn.Module):
    # keep this for ablations / classifier baseline
    def __init__(self, vocab_size=4096, d_txt=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_txt)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.pad = 0

    def tokenize(self, questions, max_len=32):
        toks = []
        for q in questions:
            ws = q.strip().lower().split()[:max_len]
            ids = [ (hash(w) % (self.emb.num_embeddings-1)) + 1 for w in ws ] or [1]
            ids += [0]*(max_len-len(ids))
            toks.append(ids)
        return torch.tensor(toks, dtype=torch.long)

    def forward(self, token_ids):
        emb = self.emb(token_ids)
        mask = (token_ids != self.pad)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (emb * mask.unsqueeze(-1)).sum(dim=1) / denom

class ClinicalBERTText(nn.Module):
    """
    Bio_ClinicalBERT pooled encoder.
    Default: freeze parameters; set fine_tune=True to unfreeze top layers.
    """
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", d_txt=768,
                 proj_dim=256, fine_tune=False):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.bert = AutoModel.from_pretrained(model_name)
        self.out_dim = d_txt  # usually 768 for BERT-base
        self.proj = nn.Linear(self.out_dim, proj_dim) if proj_dim and proj_dim != self.out_dim else nn.Identity()

        if not fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

    def tokenize(self, questions, max_len=48):
        # returns a dict of tensors
        enc = self.tok(
            list(questions),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        return enc  # {'input_ids','attention_mask','token_type_ids'(maybe)}

    def forward(self, token_batch):
        # token_batch is the dict returned by tokenize(...), already on device
        out = self.bert(**token_batch)
        # Use CLS pooling (or mean pooling of last hidden state)
        cls = out.last_hidden_state[:, 0]  # (B, hidden)
        q = self.proj(cls)                 # (B, proj_dim)
        return q

```

___

## Masked Residual Modeling (spatially aware)

`models/mrm.py`

```python
# models/mrm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.mask_token = nn.Parameter(torch.randn(1,1,c_all))

    def forward(self, feats):  # feats: (B,C,H,W)
        B,C,H,W = feats.shape
        x = self.pre(feats)
        patches = x.flatten(2).transpose(1,2)     # (B,HW,C)

        device = patches.device
        N = patches.shape[1]
        num_mask = int(self.mask_ratio * N)
        rand = torch.rand(B, N, device=device).argsort(-1)
        masked_idx, unmasked_idx = rand[:,:num_mask], rand[:,num_mask:]
        b = torch.arange(B, device=device)[:,None]

        enc_all = self.enc(patches)
        enc_unmasked = enc_all[b, unmasked_idx]
        full = self.mask_token.expand(B, N, C).clone()
        full[b, unmasked_idx] = enc_unmasked

        recon_all = self.dec(full)
        recon_masked = recon_all[b, masked_idx]
        orig_masked  = patches[b, masked_idx]

        loss = F.mse_loss(recon_masked, orig_masked)
        return {"loss_mrm": loss, "patches": patches, "recon_all": recon_all,
                "masked_idx": masked_idx, "unmasked_idx": unmasked_idx}

```

## image difference embedding (IDE) aware tiny decoder or classifer

`models/heads.py`

```python
# models/heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class IDEClassifier(nn.Module):
    """Simple closed-vocab classifier on pooled tokens; includes IDE embeddings."""
    def __init__(self, dim, num_classes):
        super().__init__()
        self.ide_ref = nn.Parameter(torch.randn(1, 1, dim))
        self.ide_cur = nn.Parameter(torch.randn(1, 1, dim))
        self.ide_abs = nn.Parameter(torch.randn(1, 1, dim))
        self.cls = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_classes)
        )

    def forward(self, sel_tokens, token_kinds=None):
        """
        sel_tokens: (B,k,Dim)
        token_kinds: (B,k) in {0:ref, 1:cur, 2:abs, 3:zone} or None
        """
        B,K,D = sel_tokens.shape
        if token_kinds is not None:
            # add learned IDEs depending on provenance
            ide = torch.zeros(B,K,D, device=sel_tokens.device)
            if (token_kinds==0).any(): ide[token_kinds==0] = self.ide_ref
            if (token_kinds==1).any(): ide[token_kinds==1] = self.ide_cur
            if (token_kinds==2).any(): ide[token_kinds==2] = self.ide_abs
            sel_tokens = sel_tokens + ide
        pooled = sel_tokens.mean(dim=1)
        return self.cls(pooled)

class TinyTransformerDecoder(nn.Module):
    """
    Optional small seq decoder (for short phrases). You can switch heads in train.py.
    """
    def __init__(self, dim=768, vocab_size=5000, nlayer=3, nhead=8, max_len=32):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim))
        dec_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim*4, batch_first=True)
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=nlayer)
        self.out = nn.Linear(dim, vocab_size)
        self.mem_proj = nn.Linear(dim, dim)

        # IDE tags (ref/cur/abs/zone) merged into memory
        self.ide_ref = nn.Parameter(torch.randn(1,1,dim))
        self.ide_cur = nn.Parameter(torch.randn(1,1,dim))
        self.ide_abs = nn.Parameter(torch.randn(1,1,dim))
        self.ide_zone= nn.Parameter(torch.randn(1,1,dim))

    def forward(self, sel_tokens, targets=None, token_kinds=None):
        B,K,D = sel_tokens.shape
        mem = self.mem_proj(sel_tokens)
        if token_kinds is not None:
            ide = torch.zeros_like(mem)
            ide[token_kinds==0] = self.ide_ref
            ide[token_kinds==1] = self.ide_cur
            ide[token_kinds==2] = self.ide_abs
            ide[token_kinds==3] = self.ide_zone
            mem = mem + ide

        if targets is None:
            # greedy decode for demo
            y = torch.zeros(B, self.max_len, dtype=torch.long, device=mem.device)
            logits_seq = []
            for t in range(self.max_len):
                q = self.tok(y) + self.pos[:, :y.size(1)]
                h = self.dec(q, mem)
                logit = self.out(h[:, -1])
                logits_seq.append(logit.unsqueeze(1))
                y[:, t] = logit.argmax(-1)
            return torch.cat(logits_seq, dim=1), y
        else:
            q = self.tok(targets) + self.pos[:, :targets.size(1)]
            h = self.dec(q, mem)
            logits = self.out(h)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0)
            return logits, loss

```

___

## Counterfactual & Regularizers

`losses.py`

```python
# losses.py
import torch
import torch.nn.functional as F

def heatmap_kl(h1, h2, eps=1e-6):
    # KL(P||Q) + KL(Q||P) symmetric
    p = (h1.float() + eps)
    q = (h2.float() + eps)
    p = p / p.sum(dim=(1,2), keepdim=True)
    q = q / q.sum(dim=(1,2), keepdim=True)
    kl1 = (p * (p.log() - q.log())).sum(dim=(1,2))
    kl2 = (q * (q.log() - p.log())).sum(dim=(1,2))
    return (kl1 + kl2).mean()

def info_nce_token_sets(toks_a, toks_b, temperature=0.07):
    """
    toks_*: (B,k,D) — mean-pool and contrast batch-wise.
    """
    a = toks_a.mean(dim=1)    # (B,D)
    b = toks_b.mean(dim=1)
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)
```

___

## Medical-Diff-VQA loader (pairs, questions, answers)

`dataset.py`

```python
# dataset.py
import csv, json
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485,0.456,0.406]
IMAGENET_STD  = [0.229,0.224,0.225]

def gray_to_rgb(img):
    if img.mode != "L": img = img.convert("L")
    return img.convert("RGB")

img_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Lambda(gray_to_rgb),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

class DiffVQADataset(Dataset):
    """
    Uses mimic_all.csv (to resolve study_id->jpg) and mimic_pair_questions.csv (pairs + QA).
    Builds closed answer vocab from the train split if not provided.
    """
    def __init__(self, data_root, pairs_csv, meta_csv, split="train", vocab=None):
        self.data_root = Path(data_root)
        self.rows = []
        self.study_to_path = self._build_map(meta_csv)

        with open(pairs_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["split"] != split: continue
                if r["study_id"] in self.study_to_path and r["ref_id"] in self.study_to_path:
                    self.rows.append(r)

        if vocab is None:
            answers = [self._norm(r["answer"]) for r in self.rows]
            itos = sorted(Counter(answers).keys())
            self.stoi = {t:i+1 for i,t in enumerate(itos)}  # 0 reserved for <pad>/<unk>
            self.itos = ["<pad>"] + itos
        else:
            self.stoi, self.itos = vocab

    def _build_map(self, meta_csv):
        m = {}
        with open(meta_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["study_id"]
                subj = row["subject_id"]
                pfx  = f"p{str(subj)[:2]}"
                pdir = self.data_root / pfx / f"p{subj}" / f"s{sid}"
                jpgs = sorted(pdir.glob("*.jpg"))
                if jpgs: m[sid] = str(jpgs[0])
        return m

    def _norm(self, s):
        return s.strip().lower().replace(".", "").replace(",", "")

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        cur_path = self.study_to_path[r["study_id"]]
        ref_path = self.study_to_path[r["ref_id"]]
        q = r["question"].strip().lower()
        a = self._norm(r["answer"])
        y = self.stoi.get(a, 0)

        img_cur = img_tf(Image.open(cur_path))
        img_ref = img_tf(Image.open(ref_path))

        return {
            "img_cur": img_cur,
            "img_ref": img_ref,
            "question": q,
            "answer_id": torch.tensor(y, dtype=torch.long),  # 0 can be <unk>
            "meta": (r["subject_id"], r["study_id"], r["ref_id"])
        }

```

___

## helpers (tokenizer wrapper, misc)

`utils.py`

```python
import logging, colorlog
from logging.handlers import RotatingFileHandler
from torch.utils.data import DataLoader
import torch


RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
COLOR_END = "\033[0m"


def collate(batch):
    # simple default collator (images are same size from transforms)
    out = {k: [] for k in batch[0].keys()}
    for b in batch:
        for k,v in b.items():
            out[k].append(v)
    out["img_cur"] = torch.stack(out["img_cur"])
    out["img_ref"] = torch.stack(out["img_ref"])
    out["answer_id"] = torch.stack(out["answer_id"])
    return out

def make_loader(ds, bs, shuffle):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=4, pin_memory=True, collate_fn=collate)


def setup_logging(log_file: str, logger_name: str = "MIMIC-CXR-JPG-DOWNLOADER"):
    # add a custom 'SUCCESS' log level
    SUCCESS_LEVEL_NUM = 25
    logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

    def success(self, message, *args, **kws):
        if self.isEnabledFor(SUCCESS_LEVEL_NUM):
            self._log(SUCCESS_LEVEL_NUM, message, args, **kws)

    logging.Logger.success = success

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # avoid adding handlers if they already exist
    if logger.hasHandlers():
        logger.handlers.clear()

    # formatter for the console with colors
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_colors={
            "DEBUG": "white",
            "INFO": "cyan",
            "SUCCESS": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_white, bg_red",
        },
    )

    # formatter for the file (plain text)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # console handler
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # this will create up to 5 backup files, each 10MB in size.
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10 * 1024 * 1024 bytes = 10MB
        backupCount=5,
        mode="a",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
```

___

## Notes

- The classifier head is a strong baseline for dataset-style answers.

- Switch to --head decoder to use the tiny IDE-aware seq decoder (you’ll want to replace the toy target creation in Stage B/C with your pseudo-difference phrases built from all_diseases.json deltas).

- The code already includes:
  - Directional residuals with calibration and alignment loss.
  - QDT+ with zone tokens, adjacency bias, gated top-k, and an evidence heatmap.
  - MRM auxiliary loss (Stage A, B, C).
  - Counterfactual evidence losses (KL on heatmaps, InfoNCE on token sets).
  - IDE tags in the heads.

___

## What’s left to plug in

### Add a tiny phrase builder from `all_diseases.json`

`phrases.py`

```python
# phrases.py
import json
from collections import defaultdict

# --- tiny helper to normalize strings
def _norm(s):
    return (s or "").strip().lower()

def load_keyinfo(path):
    """
    Reads all_diseases.json into a dict: study_id -> {'entity': {...}, 'no_entity': [...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx = {}
    for it in data:
        sid = str(it.get("study_id"))
        idx[sid] = {
            "entity": it.get("entity", {}),
            "no_entity": it.get("no_entity", []),
        }
    return idx

def _entity_descriptors(ent_dict):
    """
    ent_dict: like {'pneumothorax': {'entity_name': 'pneumothorax', 'location': [...], 'level': [...], ...}, ...}
    Returns set of canonical descriptors like:
      "pneumothorax|left|apical|minimal"
    """
    desc = set()
    for name, info in ent_dict.items():
        name = _norm(info.get("entity_name") or name)
        locs  = info.get("location") or []
        lvl   = info.get("level") or []
        # Build a few canonical combos (location x level; fallback to just name)
        if not locs and not lvl:
            desc.add(f"{name}")
        else:
            if not locs: locs = [""]
            if not lvl:  lvl  = [""]
            for L in locs:
                for V in lvl:
                    segs = [name]
                    if _norm(L): segs.append(_norm(L))
                    if _norm(V): segs.append(_norm(V))
                    desc.add("|".join(segs))
    return desc

def diff_keyinfo(cur_info, ref_info):
    """
    Returns (added, removed, changed) sets of descriptors between current and reference.
    We treat 'changed' as same disease appearing with different attributes.
    """
    cur_e = _entity_descriptors(cur_info.get("entity", {}))
    ref_e = _entity_descriptors(ref_info.get("entity", {}))
    added  = cur_e - ref_e
    removed= ref_e - cur_e

    # changed: same disease name appears on both sides but attributes differ
    # crude heuristic: match by prefix (disease name before first '|')
    cur_by_name = defaultdict(set); ref_by_name = defaultdict(set)
    for d in cur_e:
        nm = d.split("|",1)[0]
        cur_by_name[nm].add(d)
    for d in ref_e:
        nm = d.split("|",1)[0]
        ref_by_name[nm].add(d)
    changed = set()
    for nm in cur_by_name.keys() & ref_by_name.keys():
        if cur_by_name[nm] != ref_by_name[nm]:
            # flag as changed but remove from added/removed to avoid double counting
            changed.update(cur_by_name[nm] ^ ref_by_name[nm])
            added   -= cur_by_name[nm]
            removed -= ref_by_name[nm]

    return added, removed, changed

def phrase_from_delta(added, removed, changed, max_items=3):
    """
    Turn deltas into a short natural phrase (<= ~20 tokens).
    Examples:
      "new left apical minimal pneumothorax; resolved basal atelectasis"
    """
    def nice(d):
        parts = d.split("|")
        name = parts[0]
        attrs = [p for p in parts[1:] if p]
        if attrs:
            return f"{' '.join(attrs)} {name}"
        return name

    segs = []
    if added:
        a = "; ".join(nice(x) for x in list(added)[:max_items])
        segs.append(f"new {a}")
    if removed:
        r = "; ".join(nice(x) for x in list(removed)[:max_items])
        segs.append(f"resolved {r}")
    if changed:
        c = "; ".join(nice(x) for x in list(changed)[:max_items])
        segs.append(f"changed {c}")

    if not segs:
        return "no significant change"
    return "; ".join(segs)

def build_diff_phrase(study_id_cur, study_id_ref, keyinfo_index):
    """
    Public API: given current/ref study_ids (as str), and preloaded keyinfo_index, return phrase.
    """
    cur = keyinfo_index.get(str(study_id_cur))
    ref = keyinfo_index.get(str(study_id_ref))
    if not cur or not ref:
        return "no significant change"
    added, removed, changed = diff_keyinfo(cur, ref)
    return phrase_from_delta(added, removed, changed)
```

### Add a negated question generator

`negate.py`

```python
# negate.py
import re

_REPL = [
    # left/right
    (r"\bleft\b",  "<<<RIGHT>>>"),
    (r"\bright\b", "left"),
    (r"<<<RIGHT>>>", "right"),
    # increase/decrease / improved/worsened / new/resolved
    (r"\bincrease(d)?\b", "decrease\\1" ),
    (r"\bdecrease(d)?\b", "increase\\1" ),
    (r"\bworsen(ed|ing)?\b", "improv\\1"),
    (r"\bimprov(ed|ement|ing)?\b", "worsen\\1"),
    (r"\bnew\b", "resolved"),
    (r"\bresolved\b", "new"),
    # presence/absence
    (r"\bis there\b", "is there no"),
    (r"\bno\b", "yes"),
    (r"\byes\b", "no"),
    # higher/lower
    (r"\bhigher\b", "lower"),
    (r"\blower\b", "higher"),
]

def negate_question(q: str) -> str:
    s = q.strip().lower()
    for pat, rep in _REPL:
        s = re.sub(pat, rep, s)
    return s
```

## Full Three-stage Trainer (MRM -> Pseudo Report Warm Up -> Diff-VQA)

`train.py`

```python
# train.py
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from lib.dataset import DiffVQADataset
from lib.utils import make_loader
from lib.models import (
    DirectionalResidualStack,
    QuestionGuidedDifferenceTokenizer,
    MaskedResidualModel,
    IDEClassifier,
    TinyTransformerDecoder,
    TinyText,
    ClinicalBERTText
)
from lib.losses import heatmap_kl, info_nce_token_sets
from lib.phrases import load_keyinfo, build_diff_phrase
from lib.negate import negate_question


# --------------------------
# Args + YAML config
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to YAML config")

    # All CLI args are OPTIONAL; YAML can override them
    # Data
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--pairs_csv", type=str, default="")
    parser.add_argument("--meta_csv", type=str, default="")
    parser.add_argument("--keyinfo_json", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")

    # Model
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument(
        "--head", type=str, default="classifier", choices=["classifier", "decoder"]
    )

    # Text encoder
    parser.add_argument(
        "--text_encoder", type=str, default="tiny", choices=["tiny", "clinicalbert"]
    )
    parser.add_argument(
        "--text_model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT"
    )
    parser.add_argument("--text_finetune", action="store_true")
    parser.add_argument("--text_dim", type=int, default=768)  # ClinicalBERT hidden size
    parser.add_argument(
        "--text_proj_dim", type=int, default=256
    )  # projected dim into QDT

    # Decoder vocab (when head=decoder)
    parser.add_argument("--dec_vocab", type=int, default=6000)

    # Train hyperparams
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs_mrm", type=int, default=1)
    parser.add_argument("--epochs_warmup", type=int, default=1)
    parser.add_argument("--epochs_vqa", type=int, default=3)
    parser.add_argument("--topk", type=int, default=64)

    # Loss weights
    parser.add_argument("--lambda_mrm", type=float, default=0.1)
    parser.add_argument("--lambda_align", type=float, default=0.05)
    parser.add_argument("--lambda_cf", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # YAML overrides CLI defaults
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            setattr(args, k, v)

    # Basic sanity: these should be set by YAML or CLI
    needed = ["data_root", "pairs_csv", "meta_csv"]
    missing = [k for k in needed if not getattr(args, k, None)]
    if missing:
        raise SystemExit(
            f"Missing required settings ({', '.join(missing)}). "
            f"Provide them in the YAML passed by --config or as CLI flags."
        )

    return args


# --------------------------
# Utils
# --------------------------
def seed_all(s=42):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def text_to_ids(texts, vocab_size=6000, max_len=16):
    """Hash-based toy tokenizer for decoder targets; 0=PAD."""
    ids = []
    for t in texts:
        words = t.strip().lower().split()[:max_len] or ["<blank>"]
        row = [(hash(w) % (vocab_size - 1)) + 1 for w in words]
        row += [0] * (max_len - len(row))
        ids.append(row)
    return torch.tensor(ids, dtype=torch.long)


def tokenize_questions(text_model, batch_questions, use_hf=False, device=None):
    """Return token ids (TinyText) or dict of tensors (HF ClinicalBERT)."""
    if use_hf:
        enc = text_model.tokenize(batch_questions)
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        return enc
    else:
        return text_model.tokenize(batch_questions)


# --------------------------
# Model wrapper
# --------------------------
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
        head="classifier",
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

        # Head
        if head == "classifier":
            self.head = IDEClassifier(dim=c_all, num_classes=num_classes)
            self.is_classifier = True
        else:
            self.head = TinyTransformerDecoder(
                dim=c_all, vocab_size=num_classes, nlayer=3, nhead=8, max_len=16
            )
            self.is_classifier = False

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

        if self.is_classifier:
            logits = self.head(sel_tokens, token_kinds=None)
            return {
                "logits": logits,
                "heatmap": heatmap,
                "gate_l1": gate_l1,
                **mrm_out,
                **r,
            }
        else:
            return {
                "sel_tokens": sel_tokens,
                "heatmap": heatmap,
                "gate_l1": gate_l1,
                **mrm_out,
                **r,
            }


# --------------------------
# One epoch
# --------------------------
def run_epoch(
    stage,
    model,
    loader,
    optimizer,
    scaler,
    device,
    lambda_mrm=0.1,
    lambda_align=0.05,
    lambda_cf=0.05,
    lambda_gate=1e-3,
    classifier=True,
    vocab_size=None,
    keyinfo_idx=None,
):
    model.train()
    total_steps = len(loader)
    running = {"loss": 0.0, "acc": 0, "n": 0}

    for i, batch in enumerate(loader):
        img_cur = batch["img_cur"].to(device)
        img_ref = batch["img_ref"].to(device)
        y = batch["answer_id"].to(device)  # classifier path
        qs = [q for q in batch["question"]]
        qs_cf = [negate_question(q) for q in qs]

        tokens = tokenize_questions(
            model.text, qs, use_hf=getattr(model, "uses_hf", False), device=device
        )

        # if isinstance(tokens, torch.Tensor):
        #     tokens = tokens.to(device=device, dtype=torch.long)

        tokens_cf = tokenize_questions(
            model.text, qs_cf, use_hf=getattr(model, "uses_hf", False), device=device
        )

        # Counterfactual image swap
        img_cur_cf, img_ref_cf = img_ref, img_cur

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            out = model(img_ref, img_cur, tokens)
            out_cf = model(img_ref_cf, img_cur_cf, tokens_cf)

            # Base components
            loss_mrm = out["loss_mrm"]
            loss_align = model.drs.alignment_loss(
                out["r_pos"], out["r_neg"], out["r_abs"], out["signed"]
            )
            loss_hkl = heatmap_kl(out["heatmap"], out_cf["heatmap"])
            loss_nce = info_nce_token_sets(out["patches"], out_cf["patches"])
            loss_gate = out["gate_l1"]

            if stage == "mrm":
                loss = loss_mrm

            elif stage == "warmup":
                if classifier:
                    logits = out["logits"]
                    ce = F.cross_entropy(logits, y, ignore_index=0)
                    loss = (
                        ce
                        + lambda_mrm * loss_mrm
                        + lambda_align * loss_align
                        + lambda_gate * loss_gate
                    )
                else:
                    # Generative warm-up via KeyInfo deltas (if available)
                    phrases = []
                    for m in batch["meta"]:
                        _, sid_cur, sid_ref = m
                        phrases.append(
                            build_diff_phrase(sid_cur, sid_ref, keyinfo_idx)
                            if keyinfo_idx
                            else "no significant change"
                        )
                    targets = text_to_ids(
                        phrases, vocab_size=vocab_size, max_len=16
                    ).to(device)
                    _, loss_dec = model.head(
                        out["sel_tokens"], targets=targets, token_kinds=None
                    )
                    loss = (
                        loss_dec
                        + lambda_mrm * loss_mrm
                        + lambda_align * loss_align
                        + lambda_gate * loss_gate
                    )

            else:  # stage == "vqa"
                if classifier:
                    logits = out["logits"]
                    ce = F.cross_entropy(logits, y, ignore_index=0)
                    loss = (
                        ce
                        + lambda_mrm * loss_mrm
                        + lambda_align * loss_align
                        + lambda_cf * (loss_hkl + loss_nce)
                        + lambda_gate * loss_gate
                    )
                else:
                    # Optionally continue supervising with KeyInfo phrases
                    phrases = []
                    for m in batch["meta"]:
                        _, sid_cur, sid_ref = m
                        phrases.append(
                            build_diff_phrase(sid_cur, sid_ref, keyinfo_idx)
                            if keyinfo_idx
                            else "no significant change"
                        )
                    targets = text_to_ids(
                        phrases, vocab_size=vocab_size, max_len=16
                    ).to(device)
                    _, loss_dec = model.head(
                        out["sel_tokens"], targets=targets, token_kinds=None
                    )
                    loss = (
                        loss_dec
                        + lambda_mrm * loss_mrm
                        + lambda_align * loss_align
                        + lambda_cf * (loss_hkl + loss_nce)
                        + lambda_gate * loss_gate
                    )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running["loss"] += loss.item()
        if classifier:
            with torch.no_grad():
                pred = out["logits"].argmax(dim=-1)
                mask = y != 0
                running["acc"] += (pred[mask] == y[mask]).sum().item()
                running["n"] += mask.sum().item()

        if (i + 1) % 50 == 0:
            if classifier and running["n"] > 0:
                print(
                    f"[{stage}] {i+1}/{total_steps} "
                    f"loss={running['loss']/(i+1):.4f} acc={running['acc']/max(1,running['n']):.3f}"
                )
            else:
                print(f"[{stage}] {i+1}/{total_steps} loss={running['loss']/(i+1):.4f}")


# --------------------------
# Main
# --------------------------
def main(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets (ensure your DiffVQADataset filters by split internally)
    train_ds = DiffVQADataset(
        args.data_root, args.pairs_csv, args.meta_csv, split="train"
    )
    vocab = (train_ds.stoi, train_ds.itos)
    val_ds = DiffVQADataset(
        args.data_root, args.pairs_csv, args.meta_csv, split="val", vocab=vocab
    )

    num_classes = len(train_ds.itos) if args.head == "classifier" else args.dec_vocab
    print(f"Answer classes / Decoder vocab: {num_classes}")

    train_loader = make_loader(train_ds, args.bs, shuffle=True)
    val_loader = make_loader(
        val_ds, args.bs, shuffle=False
    )  # TODO: wire evaluation if needed

    # Model
    model = DiffVQAModel(
        backbone=args.backbone,
        text_encoder=args.text_encoder,
        text_model_name=args.text_model_name,
        text_dim=args.text_dim,
        text_proj_dim=args.text_proj_dim,
        text_finetune=args.text_finetune,
        topk=args.topk,
        num_rows=3,
        num_cols=2,
        num_classes=num_classes,
        head=args.head,
    ).to(device)

    # Optional: load CXR-CLIP / SwinTiny weights
    if args.ckpt and Path(args.ckpt).exists():
        sd = torch.load(args.ckpt, map_location="cpu")
        missing, unexpected = model.drs.backbone.load_state_dict(sd, strict=False)
        print(
            f"Loaded backbone weights: missing={len(missing)} unexpected={len(unexpected)}"
        )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # KeyInfo index for phrase supervision
    keyinfo_idx = (
        load_keyinfo(args.keyinfo_json)
        if args.keyinfo_json and Path(args.keyinfo_json).exists()
        else None
    )

    # Stage A — MaskedResidualModel warm-up
    for ep in range(int(args.epochs_mrm)):
        print(f"\n=== Stage A: MaskedResidualModel epoch {ep+1}/{args.epochs_mrm} ===")
        run_epoch(
            "mrm",
            model,
            train_loader,
            opt,
            scaler,
            device,
            lambda_mrm=1.0,
            lambda_align=0.0,
            lambda_cf=0.0,
            lambda_gate=0.0,
            classifier=(args.head == "classifier"),
        )

    # Stage B — warm-up with KeyInfo phrases
    for ep in range(int(args.epochs_warmup)):
        print(f"\n=== Stage B: Warm-up epoch {ep+1}/{args.epochs_warmup} ===")
        run_epoch(
            "warmup",
            model,
            train_loader,
            opt,
            scaler,
            device,
            lambda_mrm=0.1,
            lambda_align=0.05,
            lambda_cf=0.0,
            lambda_gate=1e-3,
            classifier=(args.head == "classifier"),
            vocab_size=(args.dec_vocab if args.head == "decoder" else None),
            keyinfo_idx=keyinfo_idx,
        )

    # Stage C — Diff-VQA finetune with counterfactual evidence losses
    for ep in range(int(args.epochs_vqa)):
        print(f"\n=== Stage C: VQA epoch {ep+1}/{args.epochs_vqa} ===")
        run_epoch(
            "vqa",
            model,
            train_loader,
            opt,
            scaler,
            device,
            lambda_mrm=args.lambda_mrm,
            lambda_align=args.lambda_align,
            lambda_cf=args.lambda_cf,
            lambda_gate=1e-3,
            classifier=(args.head == "classifier"),
            vocab_size=(args.dec_vocab if args.head == "decoder" else None),
            keyinfo_idx=keyinfo_idx,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
```

### How to run?

- Resnet50 + ClinicalBERT

    ```bash
    python train.py --config configs/clinicalbert_resnet.yaml
    ```

- Swin-Tiny + ClinicalBERT (decoder head, generative training with KeyInfo phrases):4

    ```bash
    python train.py --config configs/clinicalbert_swin_decoder.yaml
    ```

- Switch back to the tiny text stub (fast ablations) by setting:

    ```bash
    text_encoder: tiny
    text_proj_dim: 256
    ```

## Inference

`src/inference.py`

```python
import torch
import argparse
import yaml
from pathlib import Path
from PIL import Image

# Import necessary classes from your project
from lib.models import DiffVQAModel
from lib.dataset import img_tf  # Use the same image transforms

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the training configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Re-create the model architecture with the same config
    # We need the vocab from the training set to map output IDs to answers
    # This is a bit of a shortcut; ideally, you'd save the vocab with the model
    from lib.dataset import DiffVQADataset
    train_ds = DiffVQADataset(cfg['data_root'], cfg['pairs_csv'], cfg['meta_csv'], split="train")
    vocab = train_ds.itos # list of answer strings, index is the class id

    model = DiffVQAModel(
        backbone=cfg.get('backbone', 'resnet50'),
        text_encoder=cfg.get('text_encoder', 'tiny'),
        text_proj_dim=cfg.get('text_proj_dim', 256),
        topk=cfg.get('topk', 64),
        num_classes=len(vocab),
        head=cfg.get('head', 'classifier')
    ).to(device)

    # 3. Load the saved model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # 4. Prepare the inputs
    # Load and transform images
    img_ref = img_tf(Image.open(args.ref_image)).unsqueeze(0).to(device)
    img_cur = img_tf(Image.open(args.cur_image)).unsqueeze(0).to(device)
    
    # Prepare the question
    question = [args.question.strip().lower()]
    
    # Tokenize the question using the model's text encoder
    from train import tokenize_questions # We can reuse this helper
    token_batch = tokenize_questions(
        model.text, question, use_hf=getattr(model, "uses_hf", False), device=device
    )

    # 5. Run inference
    with torch.no_grad():
        output = model(img_ref, img_cur, token_batch)

    # 6. Interpret the output
    if model.is_classifier:
        logits = output['logits']
        predicted_id = logits.argmax(-1).item()
        predicted_answer = vocab[predicted_id]
        print("\n--- Inference Results ---")
        print(f"Question: {args.question}")
        print(f"Predicted Answer: {predicted_answer}")
    else:
        # Inference for a decoder model
        _, predicted_ids = model.head(output['sel_tokens'])
        # You would need a tokenizer to convert these IDs back to words
        print("\nDecoder output IDs:", predicted_ids.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained DRIFT-VQA model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training YAML config file.")
    parser.add_argument("--model_path", type=str, default="drift_vqa_final.pth", help="Path to the trained model weights (.pth).")
    parser.add_argument("--ref_image", type=str, required=True, help="Path to the reference (past) image.")
    parser.add_argument("--cur_image", type=str, required=True, help="Path to the current image.")
    parser.add_argument("--question", type=str, default="what has changed compared to the reference image?", help="The question to ask.")
    
    args = parser.parse_args()
    main(args)
```

### How to run inference

`src/inference.py`

```bash
python src/inference.py \
    --config configs/clinicalbert_resnet.yaml \
    --model_path drift_vqa_final.pth \
    --ref_image /path/to/your/past_image.jpg \
    --cur_image /path/to/your/current_image.jpg \
    --question "is there evidence of new lung opacity"
```
