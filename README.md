## Get Started

```bash
git clone --recurse-submodules git@github.com:moritzrengert/semeval2026_task_5_utn.git
cd semeval2026_task_5_utn
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Ensemble Locally

Run commands from the repository root.

### 1) Define base prediction files (tracked in `predictions/`)

```bash
DEV_PREDS=(
  predictions/dev_preds_ares.json
  predictions/dev_preds_lmms.json
  predictions/dev_preds_nli.json
  predictions/dev_preds_sbert.json
  predictions/dev_preds_sensembert.json
  predictions/dev_preds_stsbert.json
)

TEST_PREDS=(
  predictions/test_preds_ares.json
  predictions/test_preds_lmms.json
  predictions/test_preds_nli.json
  predictions/test_preds_sbert.json
  predictions/test_preds_sensembert.json
  predictions/test_preds_stsbert.json
)
```

### 2) Run average baseline

```bash
python3 src/ensemble/ensemble.py \
  --combine average \
  --dev-preds "${DEV_PREDS[@]}" \
  --test-preds "${TEST_PREDS[@]}" \
  --dev-labels-path semeval26-05-scripts/data/dev.json \
  --test-labels-path semeval26-05-scripts/data/test.json \
  --show-split-metrics \
  --out-dev predictions/dev_preds_ensemble_average.json \
  --out-test predictions/test_preds_ensemble_average.json \
  --save-meta predictions/meta_average.json
```

### 3) Run weighted ensemble (best config in current sweep)

```bash
python3 src/ensemble/ensemble.py \
  --combine weighted \
  --dev-preds "${DEV_PREDS[@]}" \
  --test-preds "${TEST_PREDS[@]}" \
  --dev-labels-path semeval26-05-scripts/data/dev.json \
  --test-labels-path semeval26-05-scripts/data/test.json \
  --weight-epochs 1800 \
  --weight-lr 0.05 \
  --weight-l2 0.001 \
  --show-split-metrics \
  --show-weights \
  --save-contributions predictions/contrib_weighted_best.json \
  --out-dev predictions/dev_preds_ensemble_best.json \
  --out-test predictions/test_preds_ensemble_best.json \
  --save-meta predictions/meta_weighted_best.json
```

### 4) Run MLP with CORAL ablation (`coral_mse`)

```bash
python3 src/ensemble/ensemble.py \
  --combine mlp \
  --dev-preds "${DEV_PREDS[@]}" \
  --test-preds "${TEST_PREDS[@]}" \
  --dev-labels-path semeval26-05-scripts/data/dev.json \
  --test-labels-path semeval26-05-scripts/data/test.json \
  --mlp-hidden-dim 16 \
  --mlp-dropout 0.0 \
  --mlp-lr 0.001 \
  --mlp-weight-decay 0.001 \
  --mlp-loss coral_mse \
  --mlp-coral-weight 1.0 \
  --mlp-mse-weight 0.1 \
  --mlp-epochs 350 \
  --mlp-batch-size 64 \
  --mlp-patience 40 \
  --mlp-progress-every 25 \
  --show-split-metrics \
  --save-contributions predictions/contrib_mlp_coral_best.json \
  --out-dev predictions/dev_preds_ensemble_mlp.json \
  --out-test predictions/test_preds_ensemble_mlp.json \
  --save-meta predictions/meta_mlp_coral_best.json
```

### 5) Optional: train NLI/SBERT experts directly

```bash
# NLI expert
python3 src/train_experts.py \
  --expert nli \
  --train-path semeval26-05-scripts/data/train.json \
  --dev-path semeval26-05-scripts/data/dev.json \
  --save-path runs/nli_best.pt

# SBERT expert
python3 src/train_experts.py \
  --expert sbert \
  --train-path semeval26-05-scripts/data/train.json \
  --dev-path semeval26-05-scripts/data/dev.json \
  --save-path runs/sbert_best.pt
```

## Ensemble Ablation Study

`Combined = (Spearman + AccWithinSD) / 2`

### Weighted Average (Sorted by Combined DESC)

| Run | Epochs | LR | L2 | Spearman | AccWithinSD | Combined | MSE | MAE | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `weighted_11` | 1800 | 0.0500 | 0.0001 | 0.595978 | 0.676344 | 0.636161 | 0.987285 | 0.837845 | 0.319355 |
| `weighted_10` | 1800 | 0.0500 | 0.0000 | 0.595975 | 0.676344 | 0.636160 | 0.987287 | 0.837845 | 0.319355 |
| `weighted_12` | 1800 | 0.0500 | 0.0010 | 0.596025 | 0.675269 | 0.635647 | 0.987270 | 0.837850 | 0.319355 |
| `weighted_09` | 1800 | 0.0300 | 0.0010 | 0.594420 | 0.672043 | 0.633232 | 0.993353 | 0.840897 | 0.313978 |
| `weighted_08` | 1800 | 0.0300 | 0.0001 | 0.594390 | 0.672043 | 0.633217 | 0.993372 | 0.840892 | 0.313978 |
| `weighted_07` | 1800 | 0.0300 | 0.0000 | 0.594390 | 0.672043 | 0.633216 | 0.993374 | 0.840892 | 0.313978 |
| `weighted_06` | 1200 | 0.0500 | 0.0010 | 0.594740 | 0.670968 | 0.632854 | 0.992009 | 0.840234 | 0.317204 |
| `weighted_04` | 1200 | 0.0500 | 0.0000 | 0.594735 | 0.670968 | 0.632851 | 0.992030 | 0.840228 | 0.317204 |
| `weighted_05` | 1200 | 0.0500 | 0.0001 | 0.594728 | 0.670968 | 0.632848 | 0.992028 | 0.840229 | 0.317204 |
| `weighted_03` | 1200 | 0.0300 | 0.0010 | 0.593618 | 0.669892 | 0.631755 | 0.998248 | 0.843512 | 0.316129 |
| `weighted_02` | 1200 | 0.0300 | 0.0001 | 0.593586 | 0.669892 | 0.631739 | 0.998266 | 0.843509 | 0.316129 |
| `weighted_01` | 1200 | 0.0300 | 0.0000 | 0.593575 | 0.669892 | 0.631734 | 0.998268 | 0.843509 | 0.316129 |

### MLP (All Losses, Sorted by Combined DESC)

| Run | Loss | Hidden | Dropout | LR | WeightDecay | CoralW | MSEW | Spearman | AccWithinSD | Combined | MSE | MAE | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp_12` | mse | 32 | 0.0 | 0.0010 | 0.0010 | 1.0 | 1.0 | 0.581893 | 0.707527 | 0.644710 | 0.978322 | 0.826162 | 0.322581 |
| `mlp_11` | mse | 32 | 0.0 | 0.0010 | 0.0001 | 1.0 | 1.0 | 0.581891 | 0.707527 | 0.644709 | 0.978305 | 0.826125 | 0.322581 |
| `mlp_16` | mse | 32 | 0.1 | 0.0010 | 0.0010 | 1.0 | 1.0 | 0.588284 | 0.698925 | 0.643604 | 0.972241 | 0.824888 | 0.336559 |
| `mlp_15` | mse | 32 | 0.1 | 0.0010 | 0.0001 | 1.0 | 1.0 | 0.588131 | 0.698925 | 0.643528 | 0.972436 | 0.824956 | 0.336559 |
| `mlp_09` | mse | 32 | 0.0 | 0.0005 | 0.0001 | 1.0 | 1.0 | 0.573732 | 0.695699 | 0.634715 | 0.990100 | 0.829316 | 0.330108 |
| `mlp_10` | mse | 32 | 0.0 | 0.0005 | 0.0010 | 1.0 | 1.0 | 0.573712 | 0.695699 | 0.634705 | 0.990087 | 0.829321 | 0.330108 |
| `mlp_13` | mse | 32 | 0.1 | 0.0005 | 0.0001 | 1.0 | 1.0 | 0.582198 | 0.683871 | 0.633034 | 0.987867 | 0.834572 | 0.329032 |
| `mlp_14` | mse | 32 | 0.1 | 0.0005 | 0.0010 | 1.0 | 1.0 | 0.582156 | 0.683871 | 0.633014 | 0.987879 | 0.834603 | 0.329032 |
| `mlp_04` | mse | 16 | 0.0 | 0.0010 | 0.0010 | 1.0 | 1.0 | 0.576881 | 0.686022 | 0.631451 | 0.986377 | 0.832602 | 0.320430 |
| `mlp_03` | mse | 16 | 0.0 | 0.0010 | 0.0001 | 1.0 | 1.0 | 0.576852 | 0.686022 | 0.631437 | 0.986436 | 0.832599 | 0.320430 |
| `mlp_02` | mse | 16 | 0.0 | 0.0005 | 0.0010 | 1.0 | 1.0 | 0.577821 | 0.683871 | 0.630846 | 0.988520 | 0.835709 | 0.317204 |
| `mlp_01` | mse | 16 | 0.0 | 0.0005 | 0.0001 | 1.0 | 1.0 | 0.577815 | 0.683871 | 0.630843 | 0.988578 | 0.835726 | 0.318280 |
| `mlp_coral_mse_04` | coral_mse | 16 | 0.0 | 0.0010 | 0.0010 | 1.0 | 0.1 | 0.549649 | 0.694624 | 0.622136 | 1.028435 | 0.845164 | 0.321505 |
| `mlp_coral_mse_03` | coral_mse | 16 | 0.0 | 0.0010 | 0.0001 | 1.0 | 0.1 | 0.549583 | 0.694624 | 0.622103 | 1.028488 | 0.845127 | 0.321505 |
| `mlp_coral_11` | coral | 32 | 0.0 | 0.0010 | 0.0001 | 1.0 | 0.0 | 0.543020 | 0.698925 | 0.620972 | 1.040032 | 0.845290 | 0.313978 |
| `mlp_coral_15` | coral | 32 | 0.1 | 0.0010 | 0.0001 | 1.0 | 0.0 | 0.540486 | 0.701075 | 0.620781 | 1.044599 | 0.849000 | 0.309677 |
| `mlp_coral_16` | coral | 32 | 0.1 | 0.0010 | 0.0010 | 1.0 | 0.0 | 0.540389 | 0.701075 | 0.620732 | 1.044653 | 0.848994 | 0.306452 |
| `mlp_coral_mse_11` | coral_mse | 32 | 0.0 | 0.0010 | 0.0001 | 1.0 | 0.1 | 0.544522 | 0.695699 | 0.620110 | 1.037783 | 0.844681 | 0.306452 |
| `mlp_coral_mse_12` | coral_mse | 32 | 0.0 | 0.0010 | 0.0010 | 1.0 | 0.1 | 0.544324 | 0.695699 | 0.620011 | 1.037941 | 0.844729 | 0.305376 |
| `mlp_coral_12` | coral | 32 | 0.0 | 0.0010 | 0.0010 | 1.0 | 0.0 | 0.542967 | 0.696774 | 0.619871 | 1.039976 | 0.845384 | 0.313978 |
| `mlp_coral_04` | coral | 16 | 0.0 | 0.0010 | 0.0010 | 1.0 | 0.0 | 0.547943 | 0.687097 | 0.617520 | 1.031577 | 0.846611 | 0.316129 |
| `mlp_coral_07` | coral | 16 | 0.1 | 0.0010 | 0.0001 | 1.0 | 0.0 | 0.547931 | 0.687097 | 0.617514 | 1.040309 | 0.855520 | 0.311828 |
| `mlp_coral_03` | coral | 16 | 0.0 | 0.0010 | 0.0001 | 1.0 | 0.0 | 0.547900 | 0.687097 | 0.617498 | 1.031599 | 0.846550 | 0.316129 |
| `mlp_coral_mse_15` | coral_mse | 32 | 0.1 | 0.0010 | 0.0001 | 1.0 | 0.1 | 0.543108 | 0.691398 | 0.617253 | 1.038309 | 0.848854 | 0.312903 |
| `mlp_coral_08` | coral | 16 | 0.1 | 0.0010 | 0.0010 | 1.0 | 0.0 | 0.547982 | 0.686022 | 0.617002 | 1.040270 | 0.855576 | 0.311828 |
| `mlp_coral_mse_16` | coral_mse | 32 | 0.1 | 0.0010 | 0.0010 | 1.0 | 0.1 | 0.543123 | 0.690323 | 0.616723 | 1.038202 | 0.848977 | 0.310753 |
| `mlp_coral_mse_08` | coral_mse | 16 | 0.1 | 0.0010 | 0.0010 | 1.0 | 0.1 | 0.548529 | 0.683871 | 0.616200 | 1.040685 | 0.855648 | 0.311828 |
| `mlp_coral_mse_07` | coral_mse | 16 | 0.1 | 0.0010 | 0.0001 | 1.0 | 0.1 | 0.548731 | 0.681720 | 0.615226 | 1.041100 | 0.855964 | 0.311828 |
| `mlp_coral_mse_10` | coral_mse | 32 | 0.0 | 0.0005 | 0.0010 | 1.0 | 0.1 | 0.529056 | 0.681720 | 0.605388 | 1.059627 | 0.859479 | 0.308602 |
| `mlp_coral_mse_09` | coral_mse | 32 | 0.0 | 0.0005 | 0.0001 | 1.0 | 0.1 | 0.529026 | 0.681720 | 0.605373 | 1.059682 | 0.859477 | 0.308602 |
| `mlp_coral_09` | coral | 32 | 0.0 | 0.0005 | 0.0001 | 1.0 | 0.0 | 0.528729 | 0.681720 | 0.605225 | 1.060249 | 0.859557 | 0.308602 |
| `mlp_coral_10` | coral | 32 | 0.0 | 0.0005 | 0.0010 | 1.0 | 0.0 | 0.528650 | 0.681720 | 0.605185 | 1.060214 | 0.859553 | 0.307527 |
| `mlp_08` | mse | 16 | 0.1 | 0.0010 | 0.0010 | 1.0 | 1.0 | 0.590979 | 0.618280 | 0.604629 | 1.114051 | 0.903360 | 0.301075 |
| `mlp_07` | mse | 16 | 0.1 | 0.0010 | 0.0001 | 1.0 | 1.0 | 0.590776 | 0.618280 | 0.604528 | 1.113582 | 0.903124 | 0.301075 |
| `mlp_coral_mse_14` | coral_mse | 32 | 0.1 | 0.0005 | 0.0010 | 1.0 | 0.1 | 0.528782 | 0.679570 | 0.604176 | 1.063436 | 0.862345 | 0.309677 |
| `mlp_coral_mse_13` | coral_mse | 32 | 0.1 | 0.0005 | 0.0001 | 1.0 | 0.1 | 0.528766 | 0.679570 | 0.604168 | 1.063403 | 0.862327 | 0.309677 |
| `mlp_coral_14` | coral | 32 | 0.1 | 0.0005 | 0.0010 | 1.0 | 0.0 | 0.528073 | 0.678495 | 0.603284 | 1.063812 | 0.862742 | 0.308602 |
| `mlp_coral_13` | coral | 32 | 0.1 | 0.0005 | 0.0001 | 1.0 | 0.0 | 0.528029 | 0.678495 | 0.603262 | 1.063846 | 0.862728 | 0.308602 |
| `mlp_06` | mse | 16 | 0.1 | 0.0005 | 0.0010 | 1.0 | 1.0 | 0.601368 | 0.602151 | 0.601759 | 1.150340 | 0.918995 | 0.302151 |
| `mlp_05` | mse | 16 | 0.1 | 0.0005 | 0.0001 | 1.0 | 1.0 | 0.601349 | 0.602151 | 0.601750 | 1.150347 | 0.918994 | 0.302151 |
| `mlp_coral_mse_01` | coral_mse | 16 | 0.0 | 0.0005 | 0.0001 | 1.0 | 0.1 | 0.532859 | 0.668817 | 0.600838 | 1.057274 | 0.864233 | 0.307527 |
| `mlp_coral_mse_02` | coral_mse | 16 | 0.0 | 0.0005 | 0.0010 | 1.0 | 0.1 | 0.532844 | 0.668817 | 0.600831 | 1.057296 | 0.864262 | 0.307527 |
| `mlp_coral_02` | coral | 16 | 0.0 | 0.0005 | 0.0010 | 1.0 | 0.0 | 0.530611 | 0.660215 | 0.595413 | 1.062409 | 0.867908 | 0.307527 |
| `mlp_coral_01` | coral | 16 | 0.0 | 0.0005 | 0.0001 | 1.0 | 0.0 | 0.530596 | 0.660215 | 0.595405 | 1.062366 | 0.867862 | 0.307527 |
| `mlp_coral_mse_06` | coral_mse | 16 | 0.1 | 0.0005 | 0.0010 | 1.0 | 0.1 | 0.533684 | 0.649462 | 0.591573 | 1.067972 | 0.873217 | 0.310753 |
| `mlp_coral_mse_05` | coral_mse | 16 | 0.1 | 0.0005 | 0.0001 | 1.0 | 0.1 | 0.533677 | 0.649462 | 0.591570 | 1.067933 | 0.873186 | 0.310753 |
| `mlp_coral_06` | coral | 16 | 0.1 | 0.0005 | 0.0010 | 1.0 | 0.0 | 0.532287 | 0.641935 | 0.587111 | 1.072404 | 0.876188 | 0.308602 |
| `mlp_coral_05` | coral | 16 | 0.1 | 0.0005 | 0.0001 | 1.0 | 0.0 | 0.532283 | 0.641935 | 0.587109 | 1.072364 | 0.876155 | 0.308602 |
