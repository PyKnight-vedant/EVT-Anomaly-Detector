# ML Security Framework: Attacks & Defenses on ML Systems

A reproducible experimental framework to evaluate major attacks on ML systems
(FGSM, PGD, Carlini & Wagner, data poisoning, backdoor attacks, membership
inference, model extraction) and corresponding defenses (adversarial
training, DP-SGD, defensive distillation, anomaly detection).

Implements Steps 1–20 of the assignment brief.

## 1. Environment Setup (Step 1)

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

GPU is optional — everything auto-detects CUDA and falls back to CPU
(`utils/seed.py::get_device`).

## 2. Project Structure (Step 2)

```
ml_security_framework/
├── config.py                 # central config: hyperparams, paths, seeds
├── requirements.txt
├── datasets/                 # downloaded data lands here (gitignored)
├── models/
│   ├── cnn.py                 # small CNN (MNIST/Fashion-MNIST)
│   └── resnet.py              # ResNet18 (CIFAR-10 / adapted input)
├── attacks/
│   ├── fgsm.py                 # Step 7
│   ├── pgd.py                  # Step 8
│   ├── carlini_wagner.py       # Step 9
│   ├── poisoning.py            # Step 10 (label flipping)
│   ├── backdoor.py             # Step 11 (trigger patch)
│   ├── membership_inference.py # Step 12 (shadow models)
│   └── model_extraction.py     # Step 13 (query + clone)
├── defenses/
│   ├── adversarial_training.py # Step 14
│   ├── dp_sgd.py                # Step 14 (Opacus)
│   ├── distillation.py          # Step 14
│   └── anomaly_detection.py     # Step 14
├── utils/
│   ├── seed.py                  # reproducibility + device
│   ├── data.py                   # Steps 3–4: download/split/preprocess
│   ├── metrics.py                 # Step 6: acc/prec/rec/F1/CM/timing/memory
│   ├── stats.py                    # Step 16: repeated runs, t-test, ANOVA
│   └── viz.py                       # Step 17: plots
├── train_baseline.py           # Step 5–6
├── run_attacks.py               # Steps 7–13 driver
├── run_defenses.py               # Step 14 driver
├── run_comparative_eval.py        # Steps 15–17 driver
├── run_full_pipeline.py            # orchestrates everything end-to-end
└── results/                         # metrics json/csv, figures, logs
```

## 3–4. Datasets & Preprocessing (Steps 3–4)

`utils/data.py` downloads MNIST, Fashion-MNIST, CIFAR-10 via `torchvision`,
splits each into 80/10/10 train/val/test, normalizes, and returns
`DataLoader`s.

```bash
python -c "from utils.data import get_dataloaders; get_dataloaders('mnist')"
```

## 5–6. Baseline Model & Evaluation (Steps 5–6)

```bash
python train_baseline.py --dataset mnist --model cnn --epochs 20
```

Trains, then reports accuracy/precision/recall/F1/confusion matrix/training
time/inference time/peak memory into `results/baseline_<dataset>.json`.

## 7–13. Attacks

```bash
python run_attacks.py --dataset mnist --model cnn --checkpoint results/baseline_mnist.pt
```

Runs FGSM (Step 7, ε ∈ {0.05, 0.1, 0.2}), PGD (Step 8), Carlini & Wagner
(Step 9), label-flipping poisoning at {5,10,20,30}% (Step 10), a backdoor
trigger attack (Step 11), membership inference via shadow models (Step 12),
and model extraction (Step 13). Results land in `results/attacks_<dataset>.json`.

## 14. Defenses

```bash
python run_defenses.py --dataset mnist --model cnn
```

Runs adversarial training, DP-SGD (Opacus), defensive distillation, and an
anomaly-detection filter for poisoned samples, evaluating each against the
attacks above.

## 15–17. Comparative Evaluation, Statistics, Visualization

```bash
python run_comparative_eval.py --dataset mnist --seeds 5
```

Repeats each experiment over 5 seeds (Step 16: mean ± std, paired t-test /
one-way ANOVA), aggregates attack vs. defense results (Step 15), and saves
confusion matrices, ROC curves, robustness curves, attack-success bar
charts, and runtime comparisons to `results/figures/` (Step 17).

## 18. Reproducibility (Step 18)

`config.py` fixes seeds for `random`/`numpy`/`torch`/CUDA, and every run
logs library versions, hyperparameters, and git commit (if available) into
`results/run_manifest_<timestamp>.json` via `utils/seed.py::log_manifest`.

## End-to-end (Steps 1–17 in one command)

```bash
python run_full_pipeline.py --dataset mnist --model cnn --epochs 20 --seeds 5
```

## 19–20. Discussion & Report

`results/comparative_summary.csv` plus the generated figures give the raw
material for the trade-off discussion (security vs. privacy vs. compute vs.
accuracy) and the final report sections (Introduction … Conclusion) called
for in Steps 19–20 — write-up is left to you since it's project-specific.

## Notes on scale

Default `--epochs` and PGD/CW iteration counts are tuned to actually finish
on a laptop CPU in reasonable time. Bump them up (`--epochs 30`, more PGD
steps, more CW steps/binary-search) for a stronger CPU/GPU box or for final
report numbers.

## Revision additions (EVT validation, multi-seed statistics)

```bash
# 1. Defenses over 5 seeds (+ real DP-SGD epsilon sweep). ~20 min/seed on a laptop GPU.
python run_defenses.py --dataset mnist --model cnn --epochs 10 --seeds 5 --dp_sweep
# 2. EVT (GPD) vs empirical-quantile calibration study (needs results/checkpoints/baseline_*.pt)
python run_evt_calibration.py --dataset mnist --model cnn
# 3. Figures + paired t-tests / Cohen's d / 95% CI / ANOVA
python run_comparative_eval.py --dataset mnist --model cnn --epochs 30 --seeds 5
```

* EVT / empirical thresholds are now calibrated on the **clean validation split**
  (`ref_loader`), then applied to the poisoned training data. The old
  fit-on-contaminated-data behaviour is kept as `*_evt_uncalibrated` (ablation).
* alpha sweep (0.001 ... 0.05), GPD vs. empirical quantile, in
  `defenses_*.json` (`*_alpha_sweep`, `evt_evasion_detection.alpha_sweep`).
* Evasion detector is tested on FGSM 0.1/0.2, PGD 0.2/40 and C&W (+ AUC, ROC).
* Per-seed results: `results/defenses_multiseed_*.json`; statistics end up in
  `results/comparative_summary_*.json` under `defense_stats`.
* All figures are drawn from real results; the old synthetic placeholder plots
  were removed. Delete stale `results/figures/{training_curves,grouped_bar}_*.png`.
"# EVT-Anomaly-Detector" 
