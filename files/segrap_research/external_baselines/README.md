# External Baseline Study

(Paper section name: **"Comparison with External Medical Segmentation Baselines"**.
Do NOT call this "Stage6" or "Stage7" anywhere in the write-up — Stage6 domain
generalisation is deferred, and this directory is unrelated to that line of work.)

## 1. Purpose

Stage5C (Fixed β=0.05 FFT-main residual frequency fusion) has converged as the
current final candidate from the Stage1–5 ablation series:

| Run | Params | Dice | HD95 | SDice@1mm | SDice@2mm |
|---|---|---|---|---|---|
| Stage5C seed2 | 29.678822M | 0.8493 | 0.8241 | 0.9591 | 0.9869 |
| Stage5C seed3 | 29.678822M | 0.842985 | 0.822497 | 0.957164 | 0.985868 |

Stage5C has strong statistical support over the full-DWT AFS-DSN baseline
(especially on HD95/Surface Dice), but only a small, **not statistically
significant** numerical edge over the FcaNet baseline. What is missing is not
another internal ablation — it's a comparison against widely-used, independent
3D medical segmentation architectures, so a reader can judge where Stage5C sits
relative to common practice, not just relative to its own ablation family.

## 2. Why Stage6 is deferred

Stage6 (domain generalisation) is a separate, larger research question. Running
it now would dilute focus before the more immediately decision-relevant question
("does Stage5C hold up against standard external baselines on the same task?")
is answered. Stage6 is paused, not cancelled.

## 3. Baselines in this study

| Name (use this in the paper) | Source | Status |
|---|---|---|
| 3D U-Net (or "nnU-Net-style 3D U-Net") | MONAI `UNet` | implemented |
| SegResNet | MONAI `SegResNet` | implemented |
| MedNeXt-S | `nnunet_mednext` (pip: `mednextv1`, from MIC-DKFZ/MedNeXt) | implemented |
| True nnU-Net (`nnunetv2` framework) | official nnU-Net | **feasibility-only, not trained** — see §9 |

## 4. Fair comparison rules (do not violate these)

- Dataset split: unchanged (`data/segrap_dataset.py`'s existing train/val/test `case_id` assignment).
- `THIN_WALL_OARS`: unchanged, same 10 OARs.
- Crop / resize / preprocessing / cache reading: unchanged — every baseline goes
  through the exact same `SegRapDataset(..., mode='thin_wall', cache_dir=...)`.
- Evaluation metrics: unchanged — every baseline's test CSV comes from the same
  `utils.metrics.evaluate_dataset()` Stage5C uses, called with zero modification.
- Results CSV columns: identical to Stage5C —
  `oar_name, is_thin_wall, dice, iou, hd95, surface_dice_1mm, surface_dice_2mm, case_id`
  (this is the literal column order `evaluate_dataset` produces; not hand-typed,
  so it cannot silently drift from Stage5C's format).
- Train/val/test flow: identical to `stage5_moe_router/train_fft_residual.py`
  (same `train_one_epoch` / `validate` / `evaluate_dataset` calls, same
  checkpoint-selection criterion — best **val Dice** — same scheduler). Only the
  model backbone differs.
- Loss: CE + Dice (`models.losses.CombinedLoss`), no FFL, no frequency branch, no
  attention module added to any external baseline.
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-5. Scheduler: `WarmupCosineScheduler`
  (same as Stage5C; not changed for any baseline).
- Stage5C's own results/checkpoints are never re-run or modified by this study.

## 5. Naming discipline — read before writing the paper

- **MONAI `UNet` is NOT true nnU-Net.** Call it **"3D U-Net"** or
  **"nnU-Net-style 3D U-Net"**. nnU-Net the framework does its own automated
  preprocessing, architecture/patch-size planning, and training loop — none of
  which is used here (this study deliberately reuses Stage5C's existing crop
  pipeline instead, for fairness — see §4).
- **"True nnU-Net"** may only be claimed if the official `nnunetv2` framework
  was actually run end-to-end. As of this study, it has not been — see
  `nnunet_feasibility.py` and §9 below for why, and what it would take.
- Do not write "Stage5C significantly outperforms FcaNet" — that comparison did
  not reach statistical significance (see project notes). Acceptable framing:
  "Stage5C remained competitive with FcaNet and showed small numerical
  improvements across two seeds, while providing statistically supported
  boundary improvements over the full DWT baseline."

## 6. How to run the smoke test

```bash
cd segrap_research   # or /workspace/segrap_research on RunPod

python external_baselines/smoke_test.py --model_name unet
python external_baselines/smoke_test.py --model_name segresnet
python external_baselines/smoke_test.py --model_name mednext_s
python external_baselines/nnunet_feasibility.py
python external_baselines/count_params.py --all --output ./results/external_baselines/external_model_params.csv
```

This only checks import/forward/backward/shape/NaN — it trains nothing. Do not
proceed to §7 until every baseline you intend to train shows `[BS=1 FEASIBLE] Yes`.

## 7. How to run formal training (only after smoke tests pass)

```bash
python external_baselines/train_external.py \
  --model_name unet \
  --data_root /workspace/data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases \
  --amp --num_workers 4 --cache_dir /workspace/data/cache \
  --seed 2 --exp_name external_unet_seed2 --epochs 100 --batch_size 1 \
  --output_dir ./results/external_baselines
```

Same pattern for `--model_name segresnet` / `mednext_s`, and for `--seed 3` once
seed2 results are in (see §10 in the original task spec for the seed3 decision
rule: only add seed3 for a baseline whose seed2 result is close to or beats
Stage5C).

Outputs per run:
- `./results/external_baselines/checkpoints/<exp_name>/best.pth`
- `./results/external_baselines/logs/<exp_name>_train_log.csv`
- `./results/external_baselines/results/<exp_name>_test_results.csv`

## 8. How to do the statistical comparison

Reuse `stage3_fft_branch/statistical_analysis_fft.py` (its title still says
"DWT-baseline vs FFT branch" — that's just leftover labelling from when it was
written; the actual computation is generic and works for any two CSVs with the
same column format). Map the arguments as:

```text
--baseline = the external baseline's test_results.csv
--fft      = Stage5C's test_results.csv
```

so the printed `diff` column means `Stage5C − external_baseline`:
- Dice diff > 0 → Stage5C better
- HD95 diff < 0 → Stage5C better
- Surface Dice diff > 0 → Stage5C better

```bash
python stage3_fft_branch/statistical_analysis_fft.py \
  --baseline ./results/external_baselines/results/external_unet_seed2_test_results.csv \
  --fft      ./results/stage5c/results/s5c_fft_residual_seed2_test_results.csv \
  --out_dir  ./results/stage5c_vs_external_unet_seed2
```

Repeat for SegResNet and MedNeXt-S.

## 9. True nnU-Net feasibility (read this before considering Option A/B)

`nnunet_feasibility.py` installs no code path that trains anything; it only
reports. Two non-equivalent ways exist to add a true-nnU-Net number later:

- **Option A — binary_crop_nnunet (fair, recommended IF pursued):** feed
  nnU-Net the same pre-cropped binary OAR crops everything else here uses.
  Directly comparable task, but bypasses most of nnU-Net's own preprocessing
  automation — so it measures "nnU-Net's architecture/trainer on pre-made
  crops", not a full nnU-Net run in the way most readers assume from the name.
- **Option B — whole_volume_reference (qualitative only):** let nnU-Net run its
  own full pipeline on whole CT volumes, multi-class (bg + 10 OARs). This is
  what nnU-Net is actually for, but the task no longer matches Stage5C's
  per-OAR binary-crop setting — usable only as a non-matching-task reference,
  never in the same ablation table as Stage5C/3D-U-Net/SegResNet/MedNeXt-S.

Run `python external_baselines/nnunet_feasibility.py` for the full report
(also saved to `./results/external_baselines/nnunet_feasibility_report.txt`).
No nnU-Net training has been run as part of this study.

## 10. How to interpret the final results

See the rules agreed for this study (do not deviate from this phrasing pattern):

- **If Stage5C beats both 3D U-Net and SegResNet:**
  "Stage5C outperformed common external 3D medical segmentation baselines under
  the same thin-wall OAR binary crop setting."
- **If Stage5C beats 3D U-Net but is close to SegResNet:**
  "Stage5C achieved competitive performance against SegResNet while providing a
  frequency-aware design specifically motivated by thin-wall OAR boundary
  characteristics."
- **If SegResNet or MedNeXt-S beats Stage5C:**
  Do NOT claim SOTA. Write: "Stage5C is not the strongest universal backbone,
  but provides a compact frequency-aware architecture with systematic ablation
  evidence and strong boundary performance over the original AFS-DSN DWT
  baseline."
- **Never write:** "Stage5C significantly outperforms FcaNet" (not statistically
  significant — see §5).

## 11. Final comparison table

After all runs complete, assemble:

```text
./results/external_baselines/final_external_comparison_table.csv
```

with columns `Method,Params(M),Seed,Dice,IoU,HD95,SurfaceDice@1mm,SurfaceDice@2mm,Notes`,
including at minimum: 3D U-Net seed2, SegResNet seed2, MedNeXt-S seed2 (if
available), DWT-Full seed2, FcaNet seed2, FFT seed2, Stage5C seed2, Stage5C
seed3 (+ any seed3 external runs added per the §10/decision-rule above).

## 12. Known environment notes (this dev machine)

- `monai`, `nnunetv2`, and `mednextv1` (importable as `nnunet_mednext`) were not
  preinstalled and were installed via pip for this study's smoke tests. If
  setting up a fresh RunPod box, run:
  ```bash
  pip install monai nnunetv2
  pip install git+https://github.com/MIC-DKFZ/MedNeXt.git
  ```
- MedNeXt-S's pip distribution name (`mednextv1`) differs from its importable
  module name (`nnunet_mednext`) — `models_external.py` already accounts for
  this; don't "fix" the import to `import mednextv1` if refactoring later.
