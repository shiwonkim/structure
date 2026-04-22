# Inference Tricks for Token BA Zero-Shot Classification

No retraining needed — same checkpoints, different inference strategies.

## 1. Patch-level voting (anchor codebook for classification) ⭐ HIGH PRIORITY

Proven in segmentation (anchor_codebook method). Apply to classification:

**Current CAP inference:**
```
tokens (B, T, D) → CAP pool → single (B, K) profile → cosine to class profiles → (B, C)
```

**Patch-level voting:**
```
Per-patch anchor sim → (P, K) per patch → cosine to each class profile → (P, C) → pool over patches → (C,)
```

Each patch independently "votes" for a class. Background patches vote weakly, discriminative patches vote strongly. Pool via mean, max, or top-k across patches.

Already implemented as `anchor_codebook` in `src/evaluation/zero_shot_segmentation.py` — just needs a classification wrapper (argmax over classes instead of spatial upsampling).

## 2. CLS + CAP ensemble

BA token layer already has a 2D CLS fallback path (`bridge_anchor_token.py:122-128`):

```
logits = α * cosine(CAP_profile, class_profiles) + (1-α) * cosine(CLS_profile, class_profiles)
```

CLS captures global semantics, CAP captures spatial structure. Different error patterns → complementary. Try α ∈ {0.3, 0.5, 0.7}.

## 3. Attention-entropy weighting

Each anchor's attention distribution has an entropy:

```
H_k = -Σ_t α_tk log(α_tk)     # attention entropy per anchor
w_k = 1 / (H_k + ε)           # inverse entropy weight
profile = w ⊙ p / ||w ⊙ p||   # reweighted, re-normalized
```

Low-entropy anchors are focused (discriminative), high-entropy are diffuse (uninformative). Upweight the discriminative ones.

## 4. Top-K patch selection per anchor

Hard-select top-K patches per anchor instead of softmax over all 257:

```
For each anchor k:
    top_K_indices = argtopk(sim[:, k], K)
    profile_k = mean(sim[top_K_indices, k])
```

Reduces background noise. Try K ∈ {16, 32, 64}.

## 5. Template-aware attention (late fusion)

Currently: each template → CAP profile → average profiles → classify.
Alternative: each template → CAP profile → classify → aggregate logits.

```
For each template t:
    profile_t = CAP(image, text_template_t)
    logit_t = cosine(profile_t, class_profiles)
prediction = max(logit_t) across templates   # or weighted mean
```

Max-pooling across templates captures template-specific discriminative signals.

## Priority ranking

1. **Patch-level voting** — highest impact, proven in segmentation
2. **CLS + CAP ensemble** — easy, cheap, likely +0.5-1 pp
3. **Attention-entropy weighting** — novel, potential on fine-grained
4. Top-K selection — marginal
5. Template-aware attention — marginal (templates are short)
