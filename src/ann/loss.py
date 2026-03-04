

import numpy as np
from src.ann.activations import softmax



def cross_entropy_loss(logits: np.ndarray, y_oh: np.ndarray) -> float:
    
    probs = np.clip(softmax(logits), 1e-12, 1.0)
    return -float(np.sum(y_oh * np.log(probs)) / logits.shape[0])


def cross_entropy_grad(logits: np.ndarray, y_oh: np.ndarray) -> np.ndarray:
    
    return (softmax(logits) - y_oh) / logits.shape[0]



def mse_loss(logits: np.ndarray, y_oh: np.ndarray) -> float:
    
    p = softmax(logits)
    return float(np.mean(np.sum((p - y_oh) ** 2, axis=1)))


def mse_grad(logits: np.ndarray, y_oh: np.ndarray) -> np.ndarray:
    
    N = logits.shape[0]
    p    = softmax(logits)
    diff = p - y_oh
    ws   = np.sum(2.0 * diff * p, axis=1, keepdims=True)
    return p * (2.0 * diff - ws) / N



_LOSS = {
    "cross_entropy": (cross_entropy_loss, cross_entropy_grad),
    "mse":           (mse_loss,           mse_grad),
}


def get_loss(name: str):
    
    key = name.strip().lower()
    if key not in _LOSS:
        raise ValueError(f"Unknown loss '{name}'. Valid: {list(_LOSS)}")
    return _LOSS[key]