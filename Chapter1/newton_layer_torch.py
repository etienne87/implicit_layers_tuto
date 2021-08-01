"""
http://implicit-layers-tutorial.org/introduction/

Newton Steps
"""
import torch
import torch.nn as nn



class TanhNewtonLayer(nn.Module):
    """
    z* = tanh(W@z* + x) : optimal z

    g(x,z) = z - tanh(W@z + x)
    dg(x,z)/dz = I - diag(sech(W@z + x).W)

    (pure gd z := dg(x,z)/dz
    """
    def __init__(self, out_features, tol=1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features)
        self.tol = tol
        self.max_iter = max_iter

    def forward(self, x):
        z = torch.tanh(x)
        self.err = float('inf')
        self.iterations = 0

        while self.err > self.tol and self.iterations < self.max_iter:
            z_linear = self.linear(z) + x
            g = z - torch.tanh(z_linear)
            self.err = torch.norm(g)

            if self.err <= self.tol:
                break

            # newton-step
            J = torch.eye(z.shape[1], device=x.device)[None,:,:] - (1 / torch.cosh(z_linear)**2)[:,:,None]*self.linear.weight[None,:,:] # jacobian N,out_features,out_features
            # what is going on here ???
            z = z - torch.solve(g[:,:,None], J)[0][:,:,0]
            self.iterations += 1

        g = z - torch.tanh(self.linear(z) + x)
        z[torch.norm(g,dim=1) > self.tol,:] = 0
        return z


def random_demo():
    layer = TanhNewtonLayer(50)
    X = torch.randn(10,50)
    Z = layer(X)
    print(f"Terminated after {layer.iterations} iterations with error {layer.err}")



if __name__ == '__main__':
    import fire;fire.Fire(random_demo)



