import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import numpy as np

from ann.neural_network    import NeuralNetwork
from ann.optimizers        import get_optimizer
from utils.data_loader     import load_data, to_onehot, get_batches
from utils.metrics         import compute_metrics

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False
    print("[Warning] wandb not installed — stdout logging only.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train NeuralNetwork on MNIST / Fashion-MNIST (NumPy only)"
    )
    p.add_argument("-d",   "--dataset",
                   default="mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("-e",   "--epochs",
                   type=int, default=15)
    p.add_argument("-b",   "--batch_size",
                   type=int, default=128)
    p.add_argument("-l",   "--loss",
                   default="mse", choices=["cross_entropy", "mse"])
    p.add_argument("-o",   "--optimizer",
                   default="rmsprop",
                   choices=["sgd", "momentum", "nag", "rmsprop"])
    p.add_argument("-lr",  "--learning_rate",
                   type=float, default=0.0016338)
    p.add_argument("-wd",  "--weight_decay",
                   type=float, default=0)
    p.add_argument("-nhl", "--num_layers",
                   type=int, default=4)
    p.add_argument("-sz",  "--hidden_size",
                   type=int, nargs="+", default=[128, 128, 64, 64])
    p.add_argument("-a",   "--activation",
                   default="relu", choices=["sigmoid", "tanh", "relu"])
    p.add_argument("-w_i", "--weight_init",
                   default="xavier", choices=["random", "xavier"])
    p.add_argument("-w_p", "--wandb_project",
                   default="da6401-assignment1")
    p.add_argument("--use_wandb",      action="store_true")
    p.add_argument("--wandb_entity",   default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--model_path",  default="src/best_model.npy")
    p.add_argument("--save_model",  action="store_true")
    p.add_argument("--save_config", default=None)
    p.add_argument("--log_grad_norms",   action="store_true")
    p.add_argument("--log_neuron_grads", action="store_true")
    p.add_argument("--num_neuron_grads", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p


# ── keep this name for autograder compatibility ──────────────────────────────
def parse_arguments():
    return build_parser().parse_args()


def load_model(model_path: str) -> dict:
    return np.load(model_path, allow_pickle=True).item()


def train(config: dict, use_wandb: bool = False) -> tuple:
    np.random.seed(config.get("seed", 42))

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_data(config["dataset"])
    y_tr_oh  = to_onehot(y_tr)
    y_val_oh = to_onehot(y_val)

    model = NeuralNetwork(config)

    opt = get_optimizer(
        config["optimizer"],
        lr           = config["learning_rate"],
        weight_decay = config.get("weight_decay", 0.0),
    )

    print(f"\n{'─'*65}")
    print(f"  Dataset={config['dataset']}  "
          f"Opt={config['optimizer']}  LR={config['learning_rate']}  "
          f"WD={config.get('weight_decay', 0)}")
    print(f"  Layers={model.hidden_sizes}  "
          f"Act={config['activation']}  Init={config.get('weight_init','xavier')}  "
          f"Loss={config['loss']}")
    print(f"{'─'*65}")

    best_f1      = -1.0
    best_weights = None
    g_iter       = 0

    for epoch in range(1, config["epochs"] + 1):

        losses = []
        for Xb, yb in get_batches(X_tr, y_tr, config["batch_size"]):
            yb_oh  = to_onehot(yb)
            logits = model.forward(Xb)
            loss   = model.compute_loss(logits, yb_oh)
            _      = model.backward(logits, yb_oh)
            opt.update(model.layers)
            losses.append(loss)

            if config.get("log_neuron_grads") and use_wandb and g_iter < 50:
                gW = model.layers[0].grad_W
                n  = config.get("num_neuron_grads", 5)
                d  = {"global_iter": g_iter}
                for ni in range(min(n, gW.shape[1])):
                    d[f"neuron_grad/n{ni}"] = float(np.linalg.norm(gW[:, ni]))
                if _WANDB:
                    wandb.log(d)
            g_iter += 1

        val_logits = model.forward(X_val)
        val_loss   = model.compute_loss(val_logits, y_val_oh)
        val_m      = compute_metrics(y_val, model.predict(X_val))

        tr_acc  = compute_metrics(y_tr, model.predict(X_tr))["accuracy"]
        tr_loss = float(np.mean(losses))

        print(f"  Ep {epoch:3d}/{config['epochs']}  "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_m['accuracy']:.4f} "
              f"val_f1={val_m['f1']:.4f}")

        if use_wandb and _WANDB:
            log = {
                "epoch":          epoch,
                "train_loss":     tr_loss,
                "train_accuracy": tr_acc,
                "val_loss":       val_loss,
                "val_accuracy":   val_m["accuracy"],
                "val_f1":         val_m["f1"],
                "val_precision":  val_m["precision"],
                "val_recall":     val_m["recall"],
            }
            if config.get("log_grad_norms"):
                for li, gn in enumerate(model.get_gradient_norms()):
                    log[f"grad_norm/layer_{li}"] = gn
            wandb.log(log)

        if val_m["f1"] > best_f1:
            best_f1      = val_m["f1"]
            best_weights = model.get_weights()

    if best_weights is not None:
        model.set_weights(best_weights)

    te_m = compute_metrics(y_te, model.predict(X_te))
    print(f"\n  [TEST]  acc={te_m['accuracy']:.4f}  "
          f"prec={te_m['precision']:.4f}  "
          f"rec={te_m['recall']:.4f}  "
          f"f1={te_m['f1']:.4f}")

    if use_wandb and _WANDB:
        wandb.log({
            "test_accuracy":  te_m["accuracy"],
            "test_precision": te_m["precision"],
            "test_recall":    te_m["recall"],
            "test_f1":        te_m["f1"],
        })

    return model, {"val": val_m, "test": te_m,
                   "hidden_sizes": model.hidden_sizes}


def main():
    args   = parse_arguments()
    config = vars(args)

    use_wandb = config.pop("use_wandb") and _WANDB
    run = None

    if use_wandb:
        run = wandb.init(
            project = config["wandb_project"],
            entity  = config.get("wandb_entity"),
            name    = config.get("wandb_run_name"),
            config  = config,
        )
        config = dict(wandb.config)

    model, results = train(config, use_wandb=use_wandb)

    if config.get("save_model"):
        model_path = config.get("model_path", "src/best_model.npy")
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)
        np.save(model_path, model.get_weights())
        print(f"[Model] Saved → {model_path}")

    if config.get("save_config"):
        cfg_path = config["save_config"]
        os.makedirs(os.path.dirname(cfg_path) if os.path.dirname(cfg_path) else ".", exist_ok=True)
        out = {
            "dataset":       config.get("dataset",       "mnist"),
            "epochs":        config.get("epochs",        15),
            "batch_size":    config.get("batch_size",    128),
            "loss":          config.get("loss",          "mse"),
            "optimizer":     config.get("optimizer",     "rmsprop"),
            "learning_rate": config.get("learning_rate", 0.0016338),
            "weight_decay":  config.get("weight_decay",  0),
            "num_layers":    config.get("num_layers",    4),
            "hidden_size":   results["hidden_sizes"],
            "activation":    config.get("activation",   "relu"),
            "weight_init":   config.get("weight_init",  "xavier"),
            "test_metrics":  results["test"],
        }
        with open(cfg_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[Config] Saved → {cfg_path}")

    if run:
        run.finish()


if __name__ == "__main__":
    main()