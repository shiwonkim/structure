# Server B Handoff — Repo Cleanup & Branch Integration

> **You (Claude Code on Server B) are picking up step 2 of a 4-step plan.**
> Read this whole file first. It is self-contained: it tells you the context,
> what was already done on Server A, exactly what to do here on Server B, and
> how control hands back to Server A afterward. When in doubt, **investigate the
> real git state before acting** — the commands below assume a topology you must
> verify first.

---

## 0. Project in one paragraph

This repo integrates **BridgeAnchors (BA, now also called PAL)** — a cross-modal
alignment method using K learnable anchors with Cross-Attention Pooling (CAP) —
into the **STRUCTURE** codebase, for a NeurIPS submission (14 methods × 2 encoder
scales × COCO 2014). Work happens on **two machines, Server A and Server B**,
sharing one GitHub remote (`origin`,
`https://github.com/shiwonkim/structure.git`). See `CLAUDE.md` for the full
project context; this file is only about the **cleanup + branch-integration
task**.

## 1. Why we are doing this (the mess)

- Originally both servers stayed in sync through `main`: edit on one server →
  push to `main` → pull on the other. This worked up through commit
  **`47aee4e` = "server integration v11"**.
- **After v11 the two servers stopped syncing** and each kept working
  independently. Both working trees accumulated divergent code **and a lot of
  one-off / dead / unused files** (throwaway eval scripts, debug scripts,
  exploratory experiments, temp configs, draft artifacts).
- Goal now: **clean each server's code separately on its own branch, then
  reconcile the two, then merge the final result into `main`.**

## 2. Branch topology (verify with `git fetch` first!)

As captured from Server A:

```
v1 ─ … ─ fcf4d3a (origin/serverB, OLD) ─ … ─ fc71b5e (old serverA base) ─ … ─ 47aee4e (origin/main = v11)
                                                                                   │
                                                          e1ba852 (origin/serverA = v11 + Server A cleanup)  ← step 1 DONE
```

- `origin/main` = **`47aee4e` (v11)** — the **common anchor**. Both servers'
  working trees were synced up to here, so v11 is the clean shared base.
- `origin/serverA` = **`e1ba852`** — already updated: **v11 + Server A's cleaned
  code** (step 1, done).
- `origin/serverB` = **`fcf4d3a`** — stale (an old ancestor of main, ~17 commits
  behind). You will move it forward to **v11 + Server B's cleaned code**.

**Verify on Server B before doing anything:**
```bash
git fetch origin
git rev-parse --short origin/main        # expect 47aee4e (v11)
git rev-parse --short origin/serverA     # expect e1ba852 (Server A cleaned)
git rev-parse --short origin/serverB     # expect fcf4d3a (old)
git log --oneline origin/main..HEAD      # what local work sits on top of v11?
git status --short                       # your uncommitted divergent + junk files
```

## 3. The 4-step integration plan

1. ✅ **DONE — Server A:** clean Server A's code, commit on top of v11, push to
   `serverA` branch. (`origin/serverA` = `e1ba852`.)
2. ⏳ **YOU ARE HERE — Server B:** clean Server B's code, commit **on top of v11
   (`origin/main`)**, push to `serverB` branch. **Do not touch `main`.**
3. ⬜ **Back on Server A:** `git diff serverA serverB` to see exactly how the two
   servers diverged (clean, because both sit on the v11 base), reconcile into one.
4. ⬜ **Back on Server A:** compare the reconciled result against `main`, then
   merge the final cleaned code into `main`.

**Why anchor everything on v11:** both `serverA` and `serverB` branches must be
built on the *same* base (v11) so that `git diff serverA serverB` shows **only
the post-v11 divergence + cleanup differences**, not the v1–v11 history (which is
common to both). If you instead commit onto the old `serverB` base (`fcf4d3a`),
the later diff is polluted with all the v1–v11 integration changes and becomes
useless. **Anchor on v11.**

## 4. What "cleanup" meant on Server A (mirror this judgment on Server B)

We did **not** delete anything irrecoverably for the one-offs — we **moved them
to a gitignored `_oneoff/` directory** (preserved on disk, removed from git).
Use the same pattern. What we moved / changed on Server A:

- **One-off shell scripts** → `_oneoff/scripts/`: experiment-queue waiters
  (`_*_waiter.sh`), ad-hoc re-eval drivers (`reeval_*.sh`), data/τ sweeps
  (`data_sweep_*`, `tau_sweep_*`), seg-table generators, etc. (Canonical training
  interface = `scripts/<encoder>/NN_*.sh`; canonical re-eval = `rerun_eval.py`.)
- **One-off Python scripts** → `_oneoff/scripts/`: `debug_*.py`, exploratory
  `dino_clip_*.py`, analysis experiments (`analysis_*.py`,
  `anchor_class_affinity.py`), seg-decoding ablations
  (`*_seg.py`, `sparse_seg_sweep.py`), log-parsers (`extract_cls_results.py`).
  Rule of thumb: **if nothing in the kept pipeline imports it and it's tied to a
  specific run/checkpoint/log, it's one-off.** Verify with a quick grep for
  imports before moving.
- **Dead `src/` standalone CLIs** (no importer, not an entrypoint) were *flagged*
  (e.g. `extract_features.py`, `extract_token_features.py`,
  `train_laion_addition_alignment.py`, `train_subsampled_alignment.py`,
  `zero_shot_patch_voting.py`). Keep `measure_alignment.py` (used by layer
  selection), `sail_star_mlp.py`/`siglip_loss.py`/`clip_eval_trainer.py`
  (dispatched), `dataset_preparation/prepare_*.py` (real data prep).
- **Temp configs** (`configs/_tmp_eval/`, ad-hoc eval-variant configs that differ
  from a base only in their `evaluation:` dataset list) → `_oneoff/configs/`.
- **`drafts/`** (paper draft `.tex`/`.md`/figures) → **untracked** via
  `.gitignore` + `git rm --cached -r drafts` (kept on disk).
- **`EXPERIMENT_STATUS.md`** → gitignored (overlaps `PROJECT_LOG.md`/`EXPERIMENTS.md`).
- **`docs/`** is **kept tracked** (un-ignored).
- **`configs/ba/` normalization** (do the equivalent if Server B's BA configs
  differ):
  - Unified every token-level BA config's `evaluation:` block to:
    ```yaml
    evaluation:
        token_level_zero_shot: true
        zero_shot_datasets: ["stl10","cifar100","caltech101","dtd","eurosat"]
        retrieval_datasets:  ["flickr30","coco_karpathy"]
    ```
  - CLS BA configs got the **same dataset lists but no `token_level_zero_shot`**.
  - Set the **default `pool_temperature: 0.03`** in all non-`_tau` BA token
    configs, then **deleted the pure `_tau` sweep configs** (the τ-ablation is
    done). Kept the COCO2017 ablation as `token_k512_coco2017.yaml`.

> Server B's junk will differ. Don't copy the file list blindly — **apply the
> same judgment to whatever Server B actually has.** Investigate, classify, move
> to `_oneoff/`, and keep the canonical training/eval/figure pipeline.
> `.gitignore` on this branch already ignores `_oneoff/`, `drafts/`,
> `EXPERIMENT_STATUS.md`, `archive/`, `.claude/` — confirm it carried over.

## 5. Exact git sequence for step 2 (after you finish cleaning)

This mirrors what Server A did. **Adjust only if your investigation in §2 shows a
different local state** (e.g. local `main` ahead of `origin/main` — if so, stop
and report; do not force anything).

```bash
# 0) make sure you're based on v11. Easiest: do the cleanup with your working
#    tree as-is, then commit on top of whatever your HEAD is, AS LONG AS your
#    HEAD's merge-base with origin/main is origin/main (i.e. you're at/under v11).
git fetch origin
git merge-base --is-ancestor origin/main HEAD && echo "HEAD already includes v11" \
    || echo "HEAD is at/under v11 — fine, we'll base on origin/main"

# 1) stage all cleanup (gitignored _oneoff/ drafts/ EXPERIMENT_STATUS excluded)
git add -A
git status --short | awk '{print $1}' | sort | uniq -c     # sanity-check A/M/D/R

# 2) commit on top of v11. If your current HEAD == origin/main (v11), just commit.
#    If your local main diverged from origin/main, instead create the branch from
#    origin/main first:  git stash → git checkout -B serverB origin/main → git stash pop
git commit -m "Clean up server B: prune one-off scripts/configs, mirror Server A cleanup

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"

# 3) point the serverB branch at this commit and switch to it
git checkout -B serverB

# 4) keep local main untouched (== origin/main = v11)
git branch -f main origin/main

# 5) confirm fast-forward, then push (NO force)
git merge-base --is-ancestor origin/serverB serverB && echo "ff OK" || echo "NOT ff — STOP"
git push origin serverB
```

**Do NOT push `main`.** `main` gets merged only in step 4, back on Server A.

After the push, verify:
```bash
git rev-parse --short serverB origin/serverB   # should match
git rev-parse --short main origin/main         # both 47aee4e (v11), untouched
```

## 6. Handing control back to Server A

Once `origin/serverB` holds Server B's cleaned code (v11 + cleanup), **stop on
Server B.** Everything you need is on the remote. Report back:

- the new `origin/serverB` commit hash,
- a short summary of what you moved to `_oneoff/` and any config/source changes,
- anything ambiguous you intentionally left for the human to decide.

Server A will then `git fetch`, run `git diff origin/serverA origin/serverB`, and
drive steps 3–4 (reconcile the two, then merge into `main`). Because both
branches sit on v11, that diff is clean.

## 7. Guardrails

- **Never `git push origin main`** during this task.
- **Never force-push.** Every push here should be a fast-forward; if it isn't,
  stop and report — it means the state differs from what this doc assumed.
- **Preserve, don't delete.** Move one-offs to `_oneoff/` (gitignored), don't
  `rm` them, so anything can be recovered.
- **Verify before mirroring.** Server B's files differ from Server A's; classify
  what's actually here rather than copying Server A's exact list.
- This file (`SERVER_B_HANDOFF.md`) is a transient coordination doc — it can be
  removed before the final `main` merge.
