"""Default config for cross-encoder expert (same as cross_encoder_adam).
Created by Adam Jen Khai Lo."""

MODEL_NAME = "microsoft/deberta-v3-large"
FROZEN_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
POOLING = "mean_last_3"

MAX_LENGTH = 256
SEED = 42
LR = 8e-6
BATCH_SIZE = 16
EPOCHS = 25
WEIGHT_DECAY = 0.03
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 5

LOSS_TYPE = "smooth_k2"
LOSS_X0 = 0.15
LOSS_K = 1.0
USE_FROZEN_EMBEDDINGS = True
USE_DUAL_PATH = False

LABEL_MIN = 1.0
LABEL_MAX = 5.0
LABEL_KEY = "average"

OUTPUT_DIR = "./cross_encoder_results"
PREDICTIONS_DIR = "./predictions"
