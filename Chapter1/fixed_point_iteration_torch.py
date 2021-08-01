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
    import urllib.request
    import tarfile
    import os
    import shutil
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader


    def download_mnist(url='https://www.di.ens.fr/~lelarge/MNIST.tar.gz',
                      path='/tmp/mnist/'):
        tar_name = os.path.basename(url)
        tar_filename = os.path.join(path, tar_name)
        filename = os.path.join(path, 'MNIST')
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(tar_filename):
            filedata = urllib.request.urlretrieve(url, tar_filename)
        if not os.path.exists(filename):
            shutil.unpack_archive(tar_filename, path+'/')

    path = 'Chapter1/'
    download_mnist(path=path)

    print('download mnist...')
    import os
    mnist_train = datasets.MNIST(path, train=True, download=False, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(path, train=False, download=False, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('define net...')
    # construct the simple model with fixed point layer
    import torch.optim as optim

    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(),
                          nn.Linear(784, 100),
                          TanhFixedPointLayer(100, max_iter=1),
                          nn.Linear(100, 10)
                          ).to(device)
    opt = optim.SGD(model.parameters(), lr=1e-1)
    is_fixed_point = False

    from tqdm import tqdm

    def epoch(loader, model, opt=None, monitor=None):
        total_loss, total_err, total_monitor = 0.,0.,0.
        model.eval() if opt is None else model.train()
        criterion = nn.CrossEntropyLoss()
        for X,y in tqdm(loader, leave=False):
            X,y = X.to(device), y.to(device)
            yp = model(X)
            loss = criterion(yp,y)
            if opt:
                opt.zero_grad()
                loss.backward()
                if sum(torch.sum(torch.isnan(p.grad)) for p in model.parameters()) == 0:
                  opt.step()

            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * X.shape[0]
            if monitor is not None:
                total_monitor += monitor(model)
        return total_err / len(loader.dataset), total_loss / len(loader.dataset), total_monitor / len(loader)
    for i in range(10):
        if i == 5:
            opt.param_groups[0]["lr"] = 1e-2

        train_err, train_loss, train_fpiter = epoch(train_loader, model, opt, lambda x : x[2].iterations)
        test_err, test_loss, test_fpiter = epoch(test_loader, model, monitor = lambda x : x[2].iterations)
        print(f"Train Error: {train_err:.4f}, Loss: {train_loss:.4f}, FP Iters: {train_fpiter:.2f} | " +
          f"Test Error: {test_err:.4f}, Loss: {test_loss:.4f}, FP Iters: {test_fpiter:.2f}")




if __name__ == '__main__':
    import fire;fire.Fire()



