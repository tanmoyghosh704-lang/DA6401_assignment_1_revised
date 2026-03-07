import numpy as np
from ann.activations import get_activation, softmax
from ann.loss import get_loss


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
            raise ValueError(f"Unknown weight_init '{method}'.")
        return W.astype(np.float64), np.zeros((1, self.out_size), dtype=np.float64)

    def forward(self, a_prev: np.ndarray) -> np.ndarray:
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

    def __init__(self, config):
        cfg = vars(config) if hasattr(config, '__dict__') else dict(config)

        self.activation  = cfg.get("activation",  "relu")
        self.weight_init = cfg.get("weight_init", "xavier")
        self.loss_name   = cfg.get("loss",         "cross_entropy")
        self.INPUT_SIZE  = int(cfg.get("input_size",  784))
        self.OUTPUT_SIZE = int(cfg.get("output_size", 10))

        raw      = cfg.get("hidden_size", [128, 128, 128])
        n_layers = int(cfg.get("num_layers", 3))
        if isinstance(raw, int):
            hidden = [raw] * n_layers
        elif len(raw) == 1:
            hidden = list(raw) * n_layers
        elif len(raw) == n_layers:
            hidden = list(raw)
        else:
            hidden = [raw[0]] * n_layers

        self.hidden_sizes            = hidden
        self.loss_fn, self.loss_grad = get_loss(self.loss_name)

        sizes = [self.INPUT_SIZE] + hidden + [self.OUTPUT_SIZE]
        self.layers = [
            Layer(sizes[i], sizes[i+1],
                  activation  = self.activation if i < len(sizes)-2 else "linear",
                  weight_init = self.weight_init)
            for i in range(len(sizes) - 1)
        ]

    def _parse_weights_to_pairs(self, data):
       
        if isinstance(data, dict) and "weights" in data:
            return self._parse_weights_to_pairs(data["weights"])

        
        if isinstance(data, dict):
            pairs = []
            i = 1
            while f"W{i}" in data:
                pairs.append((np.array(data[f"W{i}"], dtype=np.float64),
                               np.array(data[f"b{i}"], dtype=np.float64)))
                i += 1
            if pairs:
                return pairs
            return None

        if not isinstance(data, (list, tuple)):
            return None

        if len(data) == 0:
            return None

        first = data[0]

        
        if isinstance(first, dict) and "W" in first:
            return [(np.array(w["W"], dtype=np.float64),
                     np.array(w["b"], dtype=np.float64)) for w in data]

        
        if isinstance(first, (list, tuple, np.ndarray)) and not isinstance(first, np.ndarray):
            if len(first) == 2:
                return [(np.array(w[0], dtype=np.float64),
                         np.array(w[1], dtype=np.float64)) for w in data]

        
        if isinstance(first, np.ndarray) and len(data) % 2 == 0:
            pairs = []
            for i in range(0, len(data), 2):
                W = np.array(data[i],   dtype=np.float64)
                b = np.array(data[i+1], dtype=np.float64)
                if W.ndim == 2 and (b.ndim == 1 or b.ndim == 2):
                    pairs.append((W, b))
                else:
                    
                    return None
            return pairs

        
        if isinstance(first, np.ndarray) and first.ndim == 2:
            pairs = []
            for W in data:
                W = np.array(W, dtype=np.float64)
                b = np.zeros((1, W.shape[1]), dtype=np.float64)
                pairs.append((W, b))
            return pairs

        return None

    def set_weights(self, data):
        """Robust set_weights — handles every format and rebuilds layer sizes."""
        pairs = self._parse_weights_to_pairs(data)
        if pairs is None or len(pairs) == 0:
            return

        
        new_layers = []
        for i, (W, b) in enumerate(pairs):
            in_s, out_s = W.shape[0], W.shape[1]
            act = "linear" if i == len(pairs) - 1 else self.activation
            layer = Layer(in_s, out_s, activation=act, weight_init=self.weight_init)
            layer.W     = W
            layer.b     = b.reshape(1, out_s)
            layer.grad_W = np.zeros_like(W)
            layer.grad_b = np.zeros_like(layer.b)
            new_layers.append(layer)

        self.layers      = new_layers
        self.INPUT_SIZE  = new_layers[0].in_size
        self.OUTPUT_SIZE = new_layers[-1].out_size
        self.hidden_sizes = [l.out_size for l in new_layers[:-1]]

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

    def get_hidden_activations(self) -> list:
        return [l._a for l in self.layers[:-1]]

    def get_gradient_norms(self) -> list:
        return [float(np.linalg.norm(l.grad_W)) for l in self.layers]
