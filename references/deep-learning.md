# Deep Learning — CV / NLP / audio / signals

GPU training. The winner: **a strong public backbone + smart augmentation + TTA + pseudo-labeling + an ensemble of folds/architectures**. All on Kaggle GPU kernels (T4×2 / P100), nothing locally.

## Contents
- [RECON for DL](#recon-for-dl)
- [Backbone and transfer learning](#backbone-and-transfer-learning)
- [Augmentation](#augmentation)
- [Cross-validation and OOF in DL](#cross-validation-and-oof-in-dl)
- [TTA](#tta-test-time-augmentation)
- [Pseudo-labeling](#pseudo-labeling)
- [Ensembling](#ensembling)
- [Kernel infrastructure](#kernel-infrastructure)

---

## RECON for DL

- Discussion + public notebooks: which backbone/resolution/loss the leaders use.
- Past competitions in the same domain (arsenal.md → farid.one): segmentation, detection, classification — methods transfer almost verbatim.
- External data: is it allowed? Often the #1 boost. Check the rules.

## Backbone and transfer learning

- Don't train from scratch. Take a pretrained backbone (timm for CV: EfficientNet, ConvNeXt, ViT; HF Transformers for NLP: DeBERTa-v3, RoBERTa, etc.).
- Mirror the leaders' choice from discussion, then widen the diversity: 2–3 different architectures → their ensemble beats one tuned model.
- Progressive: smaller resolution/model for fast iteration, larger for the final.
- Upload backbone weights as a private dataset (internet in a submission kernel is often off).

## Augmentation

- CV: flip, crop, scale, rotate; **Mixup / CutMix** — almost always a boost on classification; RandAugment.
- Stronger train augmentation → less overfit → but don't break semantics (don't flip text/digits where orientation is signal).
- NLP: backtranslation, masking, synonyms — more cautiously, easy to hurt.
- Calibrate augmentation by CV, like any hyperparameter, not "by eye."

## Cross-validation and OOF in DL

- Same principle as tabular: KFold (or Stratified/Group for leakage), save the **OOF predictions** of each fold.
- OOF are needed to (a) measure honestly, (b) feed a meta-ensemble / Hill Climbing over the DL models.
- Group split if there's leakage (one patient/object across several examples) — otherwise CV lies.

## TTA (test-time augmentation)

At inference, run several augmented versions of an example and average the predictions. Cheap, almost always +a little. Standard: original + flips (+ multi-crop / multi-scale). Measure on CV — occasionally it hurts.

## Pseudo-labeling

A powerhouse of DL competitions:
1. The best ensemble predicts the test (or an external pool).
2. Confident predictions → pseudo-labels.
3. Retrain on train + pseudo.
4. Repeat a round or two (careful with error accumulation).
Often the difference between medal/no-medal. Keep a non-pseudo version as insurance (BEST_KNOWN).

## Ensembling

- Average OOF predictions across models (different backbones × folds × seeds); weights via Hill Climbing over OOF, as in tabular.
- Segmentation/detection: average masks/logits, **WBF** (Weighted Boxes Fusion) for boxes, **NMS** across models.
- Architecture diversity > deep tuning of one model.

## Kernel infrastructure

- `enable_gpu=true`, weights/data as private dataset inputs.
- Mixed precision (AMP) — multiple times faster and bigger batches.
- Checkpoint into the kernel output; the next kernel picks it up for retraining/inference.
- Split the pipeline across kernels: train-folds (in parallel) → a separate inference/TTA/submission. Account for the 9-hour kernel limit.
- **Code competition** (hidden test, inference notebook): see `code-and-hackathon.md` — runtime, internet-off, regression.
