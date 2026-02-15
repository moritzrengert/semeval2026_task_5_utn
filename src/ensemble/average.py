from typing import Any, Dict, Tuple

import numpy as np


def combine_average(x_dev: np.ndarray, x_test: np.ndarray, n_models: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    dev_preds = np.mean(x_dev, axis=1)
    test_preds = np.mean(x_test, axis=1)
    meta = {"weights": [1.0 / n_models] * n_models}
    return dev_preds, test_preds, meta
