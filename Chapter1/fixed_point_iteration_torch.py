"""
http://implicit-layers-tutorial.org/introduction/

"""
import torch
import torch.nn as nn


class TanhFixedPointLayer(nn.Module):
    """
    z* = tanh(W@z* + x) : optimal z

    Analogy to rnn:
    - z is a form of hidden_state
    - x is never-changing input


    (A general form would be
    z* = tanh(W_h@z* + W_x@x)

    """
    def __init__(self, out_features, tol=1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features)
        self.tol = tol
        self.max_iter = max_iter

    def forward(self, x):
        z = torch.zeros_like(x)
        self.err = float('inf')
        self.iterations = 0

        while self.err > self.tol and self.iterations < self.max_iter:
            z_next = torch.tanh(self.linear(z) + x)
            self.err = torch.norm(z_next-z).item()
            self.iterations += 1
            z = z_next

        return z


def random_demo():
    layer = TanhFixedPointLayer(50)
    X = torch.randn(10,50)
    Z = layer(X)
    print(f"Terminated after {layer.iterations} iterations with error {layer.err}")


def mnist_demo():
    pass
