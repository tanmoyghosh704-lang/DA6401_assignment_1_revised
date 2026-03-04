

import numpy as np
from sklearn.model_selection import train_test_split


def load_data(dataset: str = "mnist"):
    
    name = dataset.strip().lower().replace("-", "_")

    if name == "mnist":
        from keras.datasets import mnist as _src
    elif name in ("fashion_mnist", "fashion-mnist"):
        from keras.datasets import fashion_mnist as _src
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Valid: mnist | fashion_mnist")

    (X_tr_full, y_tr_full), (X_test, y_test) = _src.load_data()

    
    X_tr_full = X_tr_full.reshape(-1, 784).astype(np.float64) / 255.0
    X_test    = X_test.reshape(-1, 784).astype(np.float64)    / 255.0

    
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full,
        test_size=0.1, random_state=42, stratify=y_tr_full,
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def to_onehot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


def get_batches(X: np.ndarray, y: np.ndarray,
                batch_size: int, shuffle: bool = True):
    
    idx = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, X.shape[0], batch_size):
        sl = idx[start: start + batch_size]
        yield X[sl], y[sl]



MNIST_CLASSES   = [str(i) for i in range(10)]
FASHION_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def get_class_names(dataset: str) -> list:
    return FASHION_CLASSES if "fashion" in dataset.lower() else MNIST_CLASSES