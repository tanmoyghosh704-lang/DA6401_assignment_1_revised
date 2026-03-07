import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import numpy as np

from ann.neural_network    import NeuralNetwork
from utils.data_loader     import load_data, get_class_names
from utils.metrics         import compute_metrics, plot_confusion_matrix

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inference with saved NeuralNetwork weights")
    p.add_argument("-d",   "--dataset",      default="mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("-e",   "--epochs",        type=int, default=15)
    p.add_argument("-b",   "--batch_size",    type=int, default=128)
    p.add_argument("-l",   "--loss",          default="cross_entropy", choices=["cross_entropy", "mse"])
    p.add_argument("-o",   "--optimizer",     default="rmsprop", choices=["sgd", "momentum", "nag", "rmsprop"])
    p.add_argument("-lr",  "--learning_rate", type=float, default=0.0016338)
    p.add_argument("-wd",  "--weight_decay",  type=float, default=0)
    p.add_argument("-nhl", "--num_layers",    type=int,   default=4)
    p.add_argument("-sz",  "--hidden_size",   type=int, nargs="+", default=[128, 128, 64, 64])
    p.add_argument("-a",   "--activation",    default="relu", choices=["sigmoid", "tanh", "relu"])
    p.add_argument("-w_i", "--weight_init",   default="xavier", choices=["random", "xavier"])
    p.add_argument("-w_p", "--wandb_project", default="da6401-assignment1")
    p.add_argument("--use_wandb",    action="store_true")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--model_path",   default="src/best_model.npy")
    p.add_argument("--config_path",  default=None)
    p.add_argument("--confusion_matrix", action="store_true")
    p.add_argument("--cm_path", default="src/confusion_matrix.png")
    p.add_argument("--seed", type=int, default=42)
    return p


def parse_arguments():
    return build_parser().parse_args()


def load_model(model_path: str) -> dict:
    return np.load(model_path, allow_pickle=True).item()


def main():
    args = parse_arguments()

    weights = load_model(args.model_path)

    
    if isinstance(weights, dict) and "config" in weights:
        saved_cfg = weights["config"]
        args.hidden_size  = saved_cfg.get("hidden_sizes", args.hidden_size)
        args.num_layers   = len(args.hidden_size)
        args.activation   = saved_cfg.get("activation",  args.activation)
        args.weight_init  = saved_cfg.get("weight_init", args.weight_init)
        args.loss         = saved_cfg.get("loss",        args.loss)

    model = NeuralNetwork(args)
    model.set_weights(weights)

    print(f"[Architecture] {model.INPUT_SIZE} → {model.hidden_sizes} → {model.OUTPUT_SIZE}")

    _, _, (X_test, y_test) = load_data(args.dataset)
    y_pred = model.predict(X_test)

    m = compute_metrics(y_test, y_pred)
    print("\n" + "=" * 52)
    print(f"  Dataset   : {args.dataset}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}  (macro)")
    print(f"  Recall    : {m['recall']:.4f}  (macro)")
    print(f"  F1 Score  : {m['f1']:.4f}  (macro)")
    print("=" * 52)

    if args.confusion_matrix:
        os.makedirs(os.path.dirname(args.cm_path) if os.path.dirname(args.cm_path) else ".", exist_ok=True)
        plot_confusion_matrix(y_test, y_pred, class_names=get_class_names(args.dataset),
                              title=f"Best Model — {args.dataset}", save_path=args.cm_path)

    if args.use_wandb and _WANDB:
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, job_type="inference")
        wandb.log({"test_accuracy": m["accuracy"], "test_precision": m["precision"],
                   "test_recall": m["recall"], "test_f1": m["f1"]})
        if args.confusion_matrix and os.path.isfile(args.cm_path):
            wandb.log({"confusion_matrix": wandb.Image(args.cm_path)})
        run.finish()


if __name__ == "__main__":
    main()
