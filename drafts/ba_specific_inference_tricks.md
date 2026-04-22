# BA-Specific Inference Tricks

Tricks that ONLY BridgeAnchors can do due to its unique anchor + CAP architecture.
Generic tricks (QE, TTA, CRF, template weighting) apply to all methods and don't
change relative comparisons — excluded here.

## Classification

### 1. Anchor-class alignment score
Standard: `sim(image_profile, class_profile)` — cosine in K-dim.
BA-specific: weight each anchor's contribution by its attention confidence.

```
For each anchor k:
    attn_k = softmax attention over image patches     # (P,)
    class_score_k = image_profile_k * class_profile_k  # scalar
    confidence_k = entropy(attn_k)                      # low = focused
Weighted classification: Σ_k (1/H_k) * class_score_k
```

Only BA has per-anchor attention patterns to compute this.

### 2. Anchor-aware template selection
Different templates activate different anchors. For each class, select the template
whose anchor activation best matches the image's anchor activation.

```
For each template t:
    text_profile_t = CAP(class_t, template_t)           # (K,)
    anchor_overlap = cosine(image_anchor_attn, text_anchor_activation)
Select template with highest overlap, classify with that template only
```

Only BA has the anchor dimension to compute this overlap.

## Retrieval

### 3. Anchor-selective similarity
Not all K anchors are relevant for every query. Use only the active anchors:

```
For query image profile p_img and candidate text profile p_txt:
    active = (p_img > threshold)  # binary mask, ~K/3 anchors
    sim = cosine(p_img[active], p_txt[active])
```

Learned sparse retrieval — only BA has semantically interpretable dimensions to mask.
Linear/MLP projections have no interpretable dimensions.

### 4. Anchor-cluster retrieval
Group K anchors into M semantic clusters (K-means on anchor vectors).
Per-cluster similarity reduces noise from individual anchor jitter.

```
Cluster anchors into M groups
Per-cluster score = mean(profile[cluster_m])
Hierarchical matching: cluster-level first, refine within clusters
```

## Segmentation

### 5. Anchor attention sharpening (τ sweep) ⭐ HIGHEST PRIORITY
Training uses τ_p=0.05 (balanced exploration). At seg inference, lower τ sharpens
attention → each anchor focuses on fewer patches → cleaner spatial boundaries.

```
# During seg inference only:
logits = sim / τ_sharp   # τ_sharp << τ_train
attn = softmax(logits, dim=patches)
```

Only BA has temperature-controlled attention pooling. Easy to sweep: {0.005, 0.01, 0.02, 0.05, 0.1}.

### 6. Per-anchor spatial maps as segmentation features
Each anchor's attention is a spatial heatmap (16×16). Stack K maps → (K, 16, 16).
This is a dense K-channel feature map. Match against per-class anchor profiles:

```
For each anchor k:
    spatial_map_k = attention weights reshaped to (16, 16)
Full feature map: (K, 16, 16)
Per-pixel class prediction: feature_map @ class_profiles.T → (C, 16, 16)
```

Different from anchor_codebook (uses similarity, not attention). Only BA produces
spatial attention maps.

### 7. Attention consensus segmentation
For each pixel, count how many anchors "agree" on the class:

```
For each patch position (i,j):
    For each anchor k:
        if attn_k[i,j] > threshold:  # anchor k "claims" this patch
            vote for class argmax(profile_k * class_profile_k)
    Majority vote across anchors
```

More robust than single-codebook lookup. Only BA has multi-anchor spatial attention.

## Priority ranking

| Trick | Task | Expected gain | Effort |
|---|---|---|---|
| **Attn sharpening (τ sweep)** | Seg | +1-3 mIoU | 5 lines |
| **Anchor-selective sim** | Retrieval | +0.5-2 pp | 15 lines |
| **Per-anchor spatial maps** | Seg | +1-2 mIoU | 20 lines |
| **Anchor-class weighting** | Cls | +0.3-1 pp | 20 lines |
| Attention consensus | Seg | +0.5-1 mIoU | 30 lines |
| Anchor-aware template | Cls | +0.3-0.5 pp | 30 lines |
