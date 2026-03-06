

import numpy as np


class BaseOptimizer:
    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0):
        self.lr           = lr
        self.weight_decay = weight_decay

    def update(self, layers: list):
        raise NotImplementedError

    def reset_state(self):
        pass



class SGD(BaseOptimizer):
    

    def update(self, layers: list):
        for l in layers:
            l.W -= self.lr * (l.grad_W + self.weight_decay * l.W)
            l.b -= self.lr * l.grad_b



class Momentum(BaseOptimizer):
    

    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.beta = beta
        self._v   = {}

    def _init(self, layer):
        lid = id(layer)
        if lid not in self._v:
            self._v[lid] = {"W": np.zeros_like(layer.W),
                            "b": np.zeros_like(layer.b)}
        return self._v[lid]

    def update(self, layers: list):
        for l in layers:
            v       = self._init(l)
            v["W"]  = self.beta * v["W"] + self.lr * (l.grad_W + self.weight_decay * l.W)
            v["b"]  = self.beta * v["b"] + self.lr * l.grad_b
            l.W    -= v["W"]
            l.b    -= v["b"]

    def reset_state(self):
        self._v = {}



class NAG(BaseOptimizer):
    

    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.beta = beta
        self._v   = {}

    def _init(self, layer):
        lid = id(layer)
        if lid not in self._v:
            self._v[lid] = {"W": np.zeros_like(layer.W),
                            "b": np.zeros_like(layer.b)}
        return self._v[lid]

    def update(self, layers: list):
        for l in layers:
            v         = self._init(l)
            vp_W      = v["W"].copy()
            vp_b      = v["b"].copy()
            v["W"]    = self.beta * v["W"] + self.lr * (l.grad_W + self.weight_decay * l.W)
            v["b"]    = self.beta * v["b"] + self.lr * l.grad_b
            l.W      -= (1.0 + self.beta) * v["W"] - self.beta * vp_W
            l.b      -= (1.0 + self.beta) * v["b"] - self.beta * vp_b

    def reset_state(self):
        self._v = {}



class RMSProp(BaseOptimizer):
    

    def __init__(self, lr=0.001, beta=0.9, eps=1e-8, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.beta = beta
        self.eps  = eps
        self._c   = {}

    def _init(self, layer):
        lid = id(layer)
        if lid not in self._c:
            self._c[lid] = {"W": np.zeros_like(layer.W),
                            "b": np.zeros_like(layer.b)}
        return self._c[lid]

    def update(self, layers: list):
        for l in layers:
            c      = self._init(l)
            gW     = l.grad_W + self.weight_decay * l.W
            gb     = l.grad_b
            c["W"] = self.beta * c["W"] + (1.0 - self.beta) * gW ** 2
            c["b"] = self.beta * c["b"] + (1.0 - self.beta) * gb ** 2
            l.W   -= self.lr * gW / (np.sqrt(c["W"]) + self.eps)
            l.b   -= self.lr * gb / (np.sqrt(c["b"]) + self.eps)

    def reset_state(self):
        self._c = {}



def get_optimizer(name: str, lr: float, weight_decay: float = 0.0,
                  **kwargs) -> BaseOptimizer:
    
    key = name.strip().lower()
    if key == "sgd":
        return SGD(lr=lr, weight_decay=weight_decay)
    elif key == "momentum":
        return Momentum(lr=lr, weight_decay=weight_decay,
                        beta=kwargs.get("beta", 0.9))
    elif key == "nag":
        return NAG(lr=lr, weight_decay=weight_decay,
                   beta=kwargs.get("beta", 0.9))
    elif key == "rmsprop":
        return RMSProp(lr=lr, weight_decay=weight_decay,
                       beta=kwargs.get("beta", 0.9),
                       eps=kwargs.get("eps", 1e-8))
    else:
        raise ValueError(
            f"Unknown optimizer '{name}'. Valid: sgd | momentum | nag | rmsprop"
        )