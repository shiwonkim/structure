# Configs

Method-by-encoder layout. Every config uses `!include ../../default.yaml` and overrides only what it needs.

## Layout

```
configs/
├── default.yaml                 ← shared defaults (our modified version; do not treat as immutable)
├── README.md                    ← this file
├── _reference_structure/        ← original STRUCTURE configs, untouched — reference only, do not run
│   ├── default_original.yaml    ← copy of default.yaml from the initial STRUCTURE commit
│   ├── ablations/
│   ├── clip/
│   ├── csa/                     ← original csa_base_*, csa_structure_*
│   ├── data/
│   ├── losses_lin/              ← original clip_base_best/last, clip_structure_{1,2,3,5}
│   ├── losses_mlp/              ← original clip_base_best/last, clip_structure_{1,2,3,5}
│   └── metrics/
├── dryrun/                      ← smoke test configs (ours)
│   ├── dryrun.yaml
│   ├── dryrun_ba.yaml
│   ├── dryrun_ba_token.yaml
│   ├── dryrun_csa.yaml
│   ├── dryrun_fa.yaml
│   ├── dryrun_mlp.yaml
│   ├── dryrun_reslowrank.yaml
│   ├── dryrun_retrieval.yaml
│   └── dryrun_structure.yaml
├── linear/<encoder>/{linear_d512.yaml, linear_d512_struct.yaml}
├── mlp/<encoder>/{mlp_d512.yaml, mlp_d512_struct.yaml}
├── csa/<encoder>/{csa_d<sim>.yaml, csa_d<sim>_struct.yaml}
├── ba/<encoder>/{cls_k128,256,512.yaml, token_k128,256,512.yaml}
└── freezealign/<encoder>/{fa_d512.yaml, fa_d512_struct.yaml}
```

`<encoder>` is one of:
- `vits_minilm` — ViT-S/14 DINOv2 (384d) + all-MiniLM-L6-v2 (384d)
- `vitl_roberta` — ViT-L/14 DINOv2 (1024d) + all-roberta-large-v1 (1024d)

## Conventions

- Each config specifies `alignment.lvm_model_name` / `alignment.llm_model_name`, `features.layer_img` / `features.layer_txt`, the alignment layer kwargs, and a minimal eval set. Everything else comes from `default.yaml`.
- CSA is dispatched via `training.cca: true` in the YAML — use the same `src/train_alignment.py` entry point as every other method.
- Token-level methods (Token BA, FreezeAlign) set `training.token_level: true` and `evaluation.token_level_zero_shot: true`.
- FreezeAlign uses `embed_dim: 512` for both vits_minilm and vitl_roberta. The earlier `input_dim == embed_dim` constraint on the shared `text_proj` head was removed by routing the CLS fallback through `local_text_proj` as a length-1 sequence (see `src/alignment/freeze_align.py`), so any `embed_dim` works.

## Adding a new encoder combo

1. Pick a tag (e.g. `vitb_clip`).
2. `cp -r configs/linear/vits_minilm configs/linear/vitb_clip` and repeat for `mlp/`, `csa/`, `ba/`, `freezealign/`.
3. In every copied file, update `alignment.lvm_model_name`, `alignment.llm_model_name`, and any dim-dependent fields (`cca_kwargs.sim_dim`, `FreezeAlignAlignmentLayer.embed_dim`).
4. Mirror the same tag under `scripts/<new_tag>/` (see `scripts/README.md`).
