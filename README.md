# DA6401 — Assignment 1: Multi-Layer Perceptron (NumPy only)

> **W&B Report:** [https://api.wandb.ai/links/tanmoyghosh704-indian-institute-of-technology-madras/2yq8u6d9]  
> **GitHub Repo:** [https://github.com/tanmoyghosh704-lang/DA6401_assignment_1_revised]

Pure-NumPy MLP for MNIST/Fashion-MNIST, with full backpropagation, 4 optimizers,
and Weights & Biases experiment tracking.

---

## Repository Structure

```
da6401_assignment_1/
├── models/.gitkeep                       
├── notebooks/
│   ├── sweep.py                   
│   └── wandb_analysis.py          
├── src/
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py         
│   │   ├── loss.py               
│   │   ├── optimizers.py          
│   │   └── neural_network.py      
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py        
│   │   └── metrics.py            
│   ├── best_model.npy            
│   ├── best_config.json          
│   ├── train.py                   
│   └── inference.py              
├── README.md
└── requirements.txt
```





## Run Order

```bash
# 0. Install
pip install -r requirements.txt

# 1. §2.1  Data Exploration
python notebooks/wandb_analysis.py --section 2.1

# 2. §2.2  Hyperparameter Sweep (≥100 runs on MNIST)
python notebooks/sweep.py --count 100

# 3. §2.3  Optimizer Showdown
python notebooks/wandb_analysis.py --section 2.3

# 4. §2.4  Vanishing Gradient
python notebooks/wandb_analysis.py --section 2.4

# 5. §2.5  Dead Neuron Investigation
python notebooks/wandb_analysis.py --section 2.5

# 6. §2.6  Loss Function Comparison
python notebooks/wandb_analysis.py --section 2.6

# 7. §2.9  Weight Init Symmetry
python notebooks/wandb_analysis.py --section 2.9

# 8. §1.1/§1.2  Train best model (update flags from your sweep results)
python src/train.py \
    -d mnist -e 10 -b 128 -l mse \
    -o rmsprop -lr 0.0016338 -wd 0.0 \
    -nhl 4 -sz 128 128 64 64 \
    -a relu -w_i xavier \
    -w_p da6401-assignment1 \
    --use_wandb --save_model --save_config src/best_config.json

# 9. §1.2/§2.8  Inference + confusion matrix
python src/inference.py \
    --model_path src/best_model.npy \
    -d mnist --confusion_matrix \
    -w_p da6401-assignment1 --use_wandb

python notebooks/wandb_analysis.py --section 2.8 --model_path src/best_model.npy

# 10. §2.10  Fashion-MNIST Transfer Challenge
python notebooks/wandb_analysis.py --section 2.10
```

---

## CLI Reference — `src/train.py` & `src/inference.py` (identical)

| Flag | Default | Description |
|------|---------|-------------|
| `-d` / `--dataset` | `mnist` | `mnist` or `fashion_mnist` |
| `-e` / `--epochs` | `15` | Training epochs |
| `-b` / `--batch_size` | `64` | Mini-batch size |
| `-l` / `--loss` | `cross_entropy` | `cross_entropy` or `mse` |
| `-o` / `--optimizer` | `rmsprop` | `sgd`, `momentum`, `nag`, `rmsprop` |
| `-lr` / `--learning_rate` | `0.001` | Learning rate |
| `-wd` / `--weight_decay` | `0.0001` | L2 regularisation |
| `-nhl` / `--num_layers` | `3` | Hidden layers |
| `-sz` / `--hidden_size` | `128 128 128` | Neurons per layer |
| `-a` / `--activation` | `relu` | `sigmoid`, `tanh`, `relu` |
| `-w_i` / `--weight_init` | `xavier` | `random`, `xavier` |
| `-w_p` / `--wandb_project` | `da6401-assignment1` |
| `--model_path` | `src/best_model.npy` | Load/save path |
