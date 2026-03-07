import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import numpy as np

from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from utils.data_loader import load_data, to_onehot, get_batches
from utils.metrics import compute_metrics

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train NeuralNetwork on MNIST (NumPy only)")
    p.add_argument("-d", "--dataset", default="mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("-e", "--epochs", type=int, default=10)
    p.add_argument("-b", "--batch_size", type=int, default=32)
    p.add_argument("-l", "--loss", default="cross_entropy", choices=["cross_entropy", "mse"])
    p.add_argument("-o", "--optimizer", default="rmsprop", choices=["sgd", "momentum", "nag", "rmsprop"])
    p.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    p.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    p.add_argument("-nhl", "--num_layers", type=int, default=3)
    
    p.add_argument("-sz", "--hidden_size", type=int, nargs="+", default=[128, 128, 128])
    p.add_argument("-a", "--activation", default="relu", choices=["sigmoid", "tanh", "relu"])
    p.add_argument("-w_i", "--weight_init", default="xavier", choices=["random", "xavier"])
    
    
    p.add_argument("--model_path", default="src/best_model.npy")
    p.add_argument("--save_model", action="store_true", default=True)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p

def parse_arguments():
    return build_parser().parse_args()

def train(config: dict, use_wandb: bool = False) -> tuple:
    np.random.seed(config.get("seed", 42))

    
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_data(config["dataset"])
    y_tr_oh = to_onehot(y_tr)
    y_val_oh = to_onehot(y_val)

    model = NeuralNetwork(config)
    opt = get_optimizer(config["optimizer"], lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0.0))

    best_f1 = -1.0
    best_weights_dict = {}

    for epoch in range(1, config["epochs"] + 1):
        losses = []
        for Xb, yb in get_batches(X_tr, y_tr, config["batch_size"]):
            yb_oh = to_onehot(yb)
            logits = model.forward(Xb)
            loss = model.compute_loss(logits, yb_oh)
            model.backward(logits, yb_oh) 
            opt.update(model.layers)
            losses.append(loss)

        
        val_preds = model.predict(X_val)
        val_m = compute_metrics(y_val, val_preds)
        tr_loss = float(np.mean(losses))

        print(f"Epoch {epoch}: Train Loss: {tr_loss:.4f} | Val Acc: {val_m['accuracy']:.4f} | Val F1: {val_m['f1']:.4f}")

        if use_wandb and _WANDB:
            wandb.log({"epoch": epoch, "train_loss": tr_loss, "val_accuracy": val_m["accuracy"], "val_f1": val_m["f1"]})

        
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            best_weights_dict = model.get_weights()  
    
    return model, best_weights_dict, compute_metrics(y_te, model.predict(X_te))

def main():
    args = parse_arguments()
    config = vars(args)

    if args.use_wandb and _WANDB:
        wandb.init(project="da6401-assignment1", config=config)

    model, best_weights, test_metrics = train(config, use_wandb=args.use_wandb)

    
    if args.save_model:
        save_dir = os.path.dirname(args.model_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        
        np.save(args.model_path, best_weights)
        print(f"[Success] Model saved to {args.model_path}")

    if _WANDB and args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()