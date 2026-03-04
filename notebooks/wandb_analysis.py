
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_cm
import wandb

from src.ann.neural_network    import NeuralNetwork
from src.ann.optimizers        import get_optimizer
from src.utils.data_loader     import load_data, to_onehot, get_batches, get_class_names
from src.utils.metrics         import compute_metrics, plot_confusion_matrix
from src.train                 import train, load_model


def _base_config(**overrides):
    """Return a dict config (mirrors best defaults from train.py)."""
    cfg = dict(
        dataset="mnist", epochs=5, batch_size=64,
        loss="cross_entropy", optimizer="rmsprop",
        learning_rate=1e-3, weight_decay=1e-4,
        num_layers=3, hidden_size=[128, 128, 128],
        activation="relu", weight_init="xavier",
        seed=42, log_grad_norms=False, log_neuron_grads=False,
    )
    cfg.update(overrides)
    return cfg


def section_2_1(project):
    """Log W&B Table: 5 sample images per MNIST class (50 total)."""
    (X_tr, y_tr), _, _ = load_data("mnist")
    run   = wandb.init(project=project, name="2.1-data-exploration-mnist")
    cols  = ["class_id", "class_name"] + [f"sample_{i}" for i in range(1, 6)]
    table = wandb.Table(columns=cols)
    for cls in range(10):
        idx  = np.where(y_tr == cls)[0][:5]
        imgs = [wandb.Image(X_tr[i].reshape(28, 28), caption=f"Digit {cls}")
                for i in idx]
        table.add_data(cls, str(cls), *imgs)
    wandb.log({"data_exploration/mnist": table})
    run.finish()
    print("[2.1] Done.")



def section_2_3(project):
    """All 4 optimizers, same arch (3×128 ReLU), first 5 epochs."""
    for opt in ["sgd", "momentum", "nag", "rmsprop"]:
        cfg = _base_config(optimizer=opt, epochs=5)
        run = wandb.init(project=project, name=f"2.3-optimizer-{opt}", config=cfg)
        train(cfg, use_wandb=True)
        run.finish()
    print("[2.3] Done.")



def section_2_4(project):
    """RMSProp, 4 hidden layers. Sigmoid vs ReLU. Log gradient norms."""
    for act in ["sigmoid", "relu"]:
        cfg = _base_config(
            optimizer="rmsprop", activation=act, epochs=10,
            num_layers=4, hidden_size=[128, 128, 128, 128],
            log_grad_norms=True,
        )
        run = wandb.init(project=project,
                         name=f"2.4-vanishing-grad-{act}", config=cfg)
        train(cfg, use_wandb=True)
        run.finish()
    print("[2.4] Done.")



def section_2_5(project):
    """ReLU + high LR vs Tanh. Log dead neuron fraction per layer."""
    (X_tr, y_tr), (X_val, y_val), _ = load_data("mnist")

    experiments = [
        dict(activation="relu", learning_rate=0.1,  name="2.5-relu-high-lr"),
        dict(activation="relu", learning_rate=1e-3, name="2.5-relu-normal-lr"),
        dict(activation="tanh", learning_rate=0.1,  name="2.5-tanh-high-lr"),
        dict(activation="tanh", learning_rate=1e-3, name="2.5-tanh-normal-lr"),
    ]

    for exp in experiments:
        name = exp.pop("name")
        cfg  = _base_config(optimizer="sgd", epochs=15, **exp)
        run  = wandb.init(project=project, name=name, config=cfg)
        np.random.seed(cfg["seed"])

        model = NeuralNetwork(cfg)
        opt   = get_optimizer(cfg["optimizer"], lr=cfg["learning_rate"])

        for epoch in range(1, cfg["epochs"] + 1):
            for Xb, yb in get_batches(X_tr, y_tr, cfg["batch_size"]):
                logits = model.forward(Xb)
                model.backward(logits, to_onehot(yb))
                opt.update(model.layers)

            # Dead neuron fraction (full training set pass)
            _    = model.forward(X_tr)
            acts = model.get_hidden_activations()
            dead = {f"dead_neurons/layer_{i}":
                    float(np.all(a == 0, axis=0).mean())
                    for i, a in enumerate(acts)}
            val_acc = compute_metrics(y_val, model.predict(X_val))["accuracy"]
            wandb.log({"epoch": epoch, "val_accuracy": val_acc, **dead})

        run.finish()
    print("[2.5] Done.")



def section_2_6(project):
    """MSE vs Cross-Entropy, identical arch and LR."""
    for loss in ["cross_entropy", "mse"]:
        cfg = _base_config(loss=loss, epochs=15)
        run = wandb.init(project=project, name=f"2.6-loss-{loss}", config=cfg)
        train(cfg, use_wandb=True)
        run.finish()
    print("[2.6] Done.")



def section_2_8(model_path, project):
    """
    1. Standard confusion matrix.
    2. Creative: top-10 confused digit pairs bar chart.
    """
    _, _, (X_te, y_te) = load_data("mnist")

    
    import argparse
    dummy_args = argparse.Namespace(
        hidden_size=[128,128,128], num_layers=3,
        activation="relu", weight_init="xavier",
        loss="cross_entropy",
    )
    model   = NeuralNetwork(dummy_args)
    weights = load_model(model_path)
    model.set_weights(weights)

    y_pred = model.predict(X_te)
    names  = get_class_names("mnist")

    run = wandb.init(project=project, name="2.8-error-analysis")

    
    os.makedirs("src", exist_ok=True)
    plot_confusion_matrix(y_te, y_pred, names,
                          title="Best Model — MNIST",
                          save_path="src/confusion_matrix.png")
    wandb.log({"confusion_matrix": wandb.Image("src/confusion_matrix.png")})

   
    cm = sk_cm(y_te, y_pred)
    np.fill_diagonal(cm, 0)
    top   = np.argsort(cm.flatten())[::-1][:10]
    rr, cc = np.unravel_index(top, cm.shape)
    labels = [f"{names[r]} → {names[c]}" for r, c in zip(rr, cc)]
    counts = [cm[r, c] for r, c in zip(rr, cc)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels[::-1], counts[::-1], color="tomato")
    ax.set_xlabel("# misclassifications")
    ax.set_title("Top-10 Confused Digit Pairs")
    fig.tight_layout()
    fig.savefig("src/top_confused_pairs.png", dpi=150)
    wandb.log({"top_confused_pairs": wandb.Image("src/top_confused_pairs.png")})

    run.finish()
    print("[2.8] Done.")



def section_2_9(project):
    """
    Zeros vs Xavier. Log per-neuron grad norms for layer 0, first 50 iters.
    Zeros → all neuron lines OVERLAP (symmetry problem).
    Xavier → lines diverge (each neuron learns independently).
    """
    for init in ["zeros", "xavier"]:
        cfg = _base_config(
            weight_init=init, optimizer="sgd", epochs=10,
            log_grad_norms=True, log_neuron_grads=True, num_neuron_grads=5,
        )
        run = wandb.init(project=project,
                         name=f"2.9-init-{init}", config=cfg)
        train(cfg, use_wandb=True)
        run.finish()
    print("[2.9] Done.")



def section_2_10(project):
    """
    3 configs from MNIST learnings applied to Fashion-MNIST.
    Budget: exactly 3 runs.
    """
    configs = [
        
        _base_config(dataset="fashion_mnist", epochs=15, optimizer="rmsprop",
                     activation="relu", weight_init="xavier",
                     learning_rate=1e-3, weight_decay=1e-4,
                     num_layers=3, hidden_size=[128, 128, 128]),
        
        _base_config(dataset="fashion_mnist", epochs=15, optimizer="momentum",
                     activation="tanh", weight_init="xavier",
                     learning_rate=1e-3, weight_decay=1e-4,
                     num_layers=4, hidden_size=[128, 128, 128, 128]),
        
        _base_config(dataset="fashion_mnist", epochs=15, optimizer="nag",
                     activation="relu", weight_init="xavier",
                     learning_rate=5e-4, weight_decay=5e-4,
                     batch_size=32,
                     num_layers=5, hidden_size=[128, 128, 128, 64, 64]),
    ]
    for i, cfg in enumerate(configs, 1):
        run = wandb.init(project=project,
                         name=f"2.10-fashion-config-{i}", config=cfg)
        train(cfg, use_wandb=True)
        run.finish()
    print("[2.10] Done.")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--section", required=True,
                   choices=["2.1","2.3","2.4","2.5","2.6","2.8","2.9","2.10"])
    p.add_argument("--project",    default="da6401-assignment1")
    p.add_argument("--model_path", default="src/best_model.npy",
                   help="Required for section 2.8")
    args = p.parse_args()

    fn = {
        "2.1":  lambda: section_2_1(args.project),
        "2.3":  lambda: section_2_3(args.project),
        "2.4":  lambda: section_2_4(args.project),
        "2.5":  lambda: section_2_5(args.project),
        "2.6":  lambda: section_2_6(args.project),
        "2.8":  lambda: section_2_8(args.model_path, args.project),
        "2.9":  lambda: section_2_9(args.project),
        "2.10": lambda: section_2_10(args.project),
    }
    fn[args.section]()


if __name__ == "__main__":
    main()