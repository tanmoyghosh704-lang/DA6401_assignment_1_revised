

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import wandb
from src.train import train



_ARCH_CONFIGS = [
    {"hidden_size": [32],                 "num_layers": 1},
    {"hidden_size": [64],                 "num_layers": 1},
    {"hidden_size": [128],                "num_layers": 1},
    {"hidden_size": [64,  64],            "num_layers": 2},
    {"hidden_size": [128, 128],           "num_layers": 2},
    {"hidden_size": [64,  64,  64],       "num_layers": 3},
    {"hidden_size": [128, 128, 128],      "num_layers": 3},
    {"hidden_size": [128, 64,  32],       "num_layers": 3},
    {"hidden_size": [128, 128, 64,  64],  "num_layers": 4},
    {"hidden_size": [128, 128, 128, 128], "num_layers": 4},
]

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        
        "dataset":          {"value": "mnist"},
        "epochs":           {"value": 10},
        "seed":             {"value": 42},
        "log_grad_norms":   {"value": False},
        "log_neuron_grads": {"value": False},
        
        "batch_size":       {"values": [32, 64, 128]},
        "loss":             {"values": ["cross_entropy", "mse"]},
        "optimizer":        {"values": ["sgd", "momentum", "nag", "rmsprop"]},
        "learning_rate":    {"distribution": "log_uniform_values",
                             "min": 1e-4, "max": 1e-1},
        "weight_decay":     {"values": [0.0, 1e-4, 5e-4, 1e-3]},
        
        "arch_index":       {"values": list(range(len(_ARCH_CONFIGS)))},
        "activation":       {"values": ["sigmoid", "tanh", "relu"]},
        "weight_init":      {"values": ["random", "xavier"]},
    },
}


def _run():
    run = wandb.init()
    cfg = dict(wandb.config)

   
    arch = _ARCH_CONFIGS[cfg.pop("arch_index", 6)]   
    cfg["hidden_size"] = arch["hidden_size"]
    cfg["num_layers"]  = arch["num_layers"]

   
    wandb.config.update({
        "hidden_size": cfg["hidden_size"],
        "num_layers":  cfg["num_layers"],
    }, allow_val_change=True)

    model, results = train(cfg, use_wandb=True)
    wandb.log({
        "test_accuracy": results["test"]["accuracy"],
        "test_f1":       results["test"]["f1"],
    })
    run.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project",  default="da6401-assignment1")
    p.add_argument("--entity",   default=None)
    p.add_argument("--count",    type=int, default=100)
    p.add_argument("--sweep_id", default=None,
                   help="Join an existing sweep instead of creating a new one")
    args = p.parse_args()

    if args.sweep_id:
        sid = args.sweep_id
        print(f"[Sweep] Joining existing sweep: {sid}")
    else:
        sid = wandb.sweep(SWEEP_CONFIG, project=args.project, entity=args.entity)
        print(f"[Sweep] Created sweep ID: {sid}")

    wandb.agent(sid, function=_run, count=args.count,
                project=args.project, entity=args.entity)


if __name__ == "__main__":
    main()