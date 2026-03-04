

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred,
                                           average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred,
                                        average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred,
                                    average="macro", zero_division=0)),
    }


def plot_confusion_matrix(y_true, y_pred, class_names=None,
                          title="Confusion Matrix", save_path=None):
    
    cm = confusion_matrix(y_true, y_pred)
    n  = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(n), yticks=np.arange(n),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig