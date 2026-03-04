

import numpy as np
from src.ann.activations import get_activation, softmax
from src.ann.loss import get_loss



class Layer:
    

    def __init__(self, in_size: int, out_size: int,
                 activation: str = "relu",
                 weight_init: str = "xavier"):
        self.in_size         = in_size
        self.out_size        = out_size
        self.activation_name = activation

        
        if activation.lower() == "linear":
            self.act_fn   = None
            self.act_grad = None
        else:
            self.act_fn, self.act_grad = get_activation(activation)

        self.W, self.b = self._init_weights(weight_init)

        
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        
        self._a_prev = None
        self._z      = None
        self._a      = None

    def _init_weights(self, method: str):
        m = method.strip().lower()
        if m == "zeros":
            W = np.zeros((self.in_size, self.out_size))
        elif m == "random":
            W = np.random.randn(self.in_size, self.out_size) * 0.01
        elif m == "xavier":
            lim = np.sqrt(6.0 / (self.in_size + self.out_size))
            W   = np.random.uniform(-lim, lim, (self.in_size, self.out_size))
        else:
            raise ValueError(
                f"Unknown weight_init '{method}'. Valid: random | xavier | zeros"
            )
        return W.astype(np.float64), np.zeros((1, self.out_size), dtype=np.float64)

    def forward(self, a_prev: np.ndarray) -> np.ndarray:
        """a_prev: (N, in_size) → (N, out_size)"""
        self._a_prev = a_prev
        self._z      = a_prev @ self.W + self.b
        self._a      = self._z if self.act_fn is None else self.act_fn(self._z)
        return self._a

    def backward(self, delta: np.ndarray) -> np.ndarray:
        
        N = self._a_prev.shape[0]
        if self.act_grad is not None:
            delta = delta * self.act_grad(self._z)
        self.grad_W = self._a_prev.T @ delta / N
        self.grad_b = np.mean(delta, axis=0, keepdims=True)
        return delta @ self.W.T



class NeuralNetwork:
    

    INPUT_SIZE  = 784
    OUTPUT_SIZE = 10

    def __init__(self, config):
        cfg = vars(config) if hasattr(config, '__dict__') else dict(config)

        self.activation  = cfg.get("activation",  "relu")
        self.weight_init = cfg.get("weight_init", "xavier")
        self.loss_name   = cfg.get("loss",         "cross_entropy")

        
        raw      = cfg.get("hidden_size", [128, 128, 128])
        n_layers = cfg.get("num_layers",  3)
        if isinstance(raw, int):
            hidden = [raw] * n_layers
        elif len(raw) == 1:
            hidden = list(raw) * n_layers
        elif len(raw) == n_layers:
            hidden = list(raw)
        else:
            print(f"[Warning] len(hidden_size)={len(raw)} != num_layers={n_layers}. "
                  "Broadcasting first value.")
            hidden = [raw[0]] * n_layers

        self.hidden_sizes            = hidden
        self.loss_fn, self.loss_grad = get_loss(self.loss_name)

        
        sizes = [self.INPUT_SIZE] + hidden + [self.OUTPUT_SIZE]
        self.layers = [
            Layer(
                sizes[i], sizes[i + 1],
                activation  = self.activation if i < len(sizes) - 2 else "linear",
                weight_init = self.weight_init,
            )
            for i in range(len(sizes) - 1)
        ]

    
    def forward(self, X: np.ndarray) -> np.ndarray:
        
        a = X.astype(np.float64)
        for layer in self.layers:
            a = layer.forward(a)
        return a

    
    def compute_loss(self, logits: np.ndarray, y_oh: np.ndarray) -> float:
        return self.loss_fn(logits, y_oh)

    
    def backward(self, logits: np.ndarray, y_oh: np.ndarray) -> list:
        
        gradients = []
        delta = self.loss_grad(logits, y_oh)          

        for layer in reversed(self.layers):
            delta = layer.backward(delta)
            gradients.append((layer.grad_W.copy(), layer.grad_b.copy()))

        return gradients   

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(softmax(self.forward(X)), axis=1)

    
    def get_weights(self) -> dict:
        
        return {
            "config": {
                "hidden_sizes": self.hidden_sizes,
                "activation":   self.activation,
                "weight_init":  self.weight_init,
                "loss":         self.loss_name,
                "input_size":   self.INPUT_SIZE,
                "output_size":  self.OUTPUT_SIZE,
            },
            "weights": [
                {"W": l.W.copy(), "b": l.b.copy()}
                for l in self.layers
            ],
        }

    def set_weights(self, data: dict):
        
        for layer, w in zip(self.layers, data["weights"]):
            layer.W = w["W"].astype(np.float64)
            layer.b = w["b"].astype(np.float64)

    
    def get_hidden_activations(self) -> list:
        
        return [l._a for l in self.layers[:-1]]

    def get_gradient_norms(self) -> list:
        
        return [float(np.linalg.norm(l.grad_W)) for l in self.layers]