

import numpy as np



def sigmoid(z: np.ndarray) -> np.ndarray:
    
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_grad(z: np.ndarray) -> np.ndarray:
    
    s = sigmoid(z)
    return s * (1.0 - s)



def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_grad(z: np.ndarray) -> np.ndarray:
    
    return 1.0 - np.tanh(z) ** 2



def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    
    return (z > 0.0).astype(np.float64)



def softmax(z: np.ndarray) -> np.ndarray:
    
    z_s = z - np.max(z, axis=1, keepdims=True)
    e   = np.exp(z_s)
    return e / np.sum(e, axis=1, keepdims=True)



_ACT = {
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh":    (tanh,    tanh_grad),
    "relu":    (relu,    relu_grad),
}


def get_activation(name: str):
    
    key = name.strip().lower()
    if key not in _ACT:
        raise ValueError(f"Unknown activation '{name}'. Valid: {list(_ACT)}")
    return _ACT[key]