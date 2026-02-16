# SemEval 2026 Task 5 UTN
This repository contains an ensemble expert system for word sense prediction (SemEval 2026 Task 5).  
It trains several base experts, exports their `dev`/`test` predictions, and combines them using ensemble methods.

Base experts:
- ARES feature expert
- LMMS feature expert
- SensEmBERT feature expert
- NLI expert
- SBERT expert
- Cross-encoder expert (`stsbert` naming in prediction files)

Ensemble methods:
- Average
- Weighted average
- Linear regression
- MLP


## Team Responsibilities
- **Noas Shaalan**: ARES/LMMS/SensEmBERT feature pipelines and CORAL MLP experts, linear-regression ensemble component.
- **Adam Jen Khai Lo**: cross-encoder expert implementation.
- **Moritz Rengert**: training/evaluation infrastructure, NLI and SBERT experts, ensemble framework (average/weighted/MLP), shared utilities.


## Setup and Installation
1. Clone and enter the repo:
```bash
git clone --recurse-submodules git@github.com:moritzrengert/semeval2026_task_5_utn.git
cd semeval2026_task_5_utn
git submodule update --init --recursive
```

2. Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Ensure annotated dataset files exist at:
- `semeval26-05-scripts/data/train.json`
- `semeval26-05-scripts/data/dev.json`
- `semeval26-05-scripts/data/test.json`

4. Feature experts require precomputed feature tensors at:
- `src/ares_expert/ares_context_features.pt`
- `src/lmms_expert/lmms_context_features.pt`
- `src/sensembert_expert/sensembert_context_features.pt`

<details>
  <summary>If missing, generate them using these instructions:</summary>
This applies only to the expert models for ARES, LMMS, and SensEmBERT.

Note: We have provided the extracted features as .pt files. However, in order to recreate these features, you must first download the pre-trained vectors from the sources listed below.

### ARES Expert
Vectors Source: https://nlp.uniroma1.it/sensembert/

To run:
1. Extract features: `python ares_expert/prep_features_final.py`
2. Train model: `python ares_expert/train_mlp_final.py`

### LMMS Expert
Vectors Source: https://figshare.com/articles/dataset/LMMS_2019_/21977219

To run:
1. Extract features: `python lmms_expert/prep_features_final.py`
2. Train model: `python lmms_expert/train_mlp_final.py`

### SensEmBERT Expert
Vectors Source: https://nlp.uniroma1.it/sensembert/

To run:
1. Extract features: `python sensembert_expert/prep_features_final.py`
2. Train model: `python sensembert_expert/train_mlp_final.py`

### General Instructions
- Ensure your environment is set up with PyTorch and the required dependencies.
- All scripts use shared utilities from the parent directory.
- Before running the training scripts, ensure the context features have been extracted or are already present in the folder.
</details>


## How to Run the Model
Run from repository root.

### 1. Train all base experts and write prediction JSON files:
```bash
python3 train_all_experts.py
```

This writes:
- `predictions/dev_preds_ares.json`
- `predictions/dev_preds_lmms.json`
- `predictions/dev_preds_nli.json`
- `predictions/dev_preds_sbert.json`
- `predictions/dev_preds_sensembert.json`
- `predictions/dev_preds_stsbert.json`
- `predictions/test_preds_*.json` for the same models

### 2. Train/evaluate all ensemble methods on those predictions:
```bash
python3 train_all_ensembles.py
```

This runs `average`, `weighted`, `linear_regression`, and `mlp`, and writes:
- `predictions/dev_preds_ensemble_<method>.json`
- `predictions/test_preds_ensemble_<method>.json`
- `predictions/meta_<method>.json`
- `predictions/contrib_<method>.json` (when available)

### 3. Optional: run a single ensemble method directly:
```bash
python3 src/ensemble/ensemble.py --combine weighted --show-contributions
```
