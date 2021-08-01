import urllib.request
import tarfile
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Chapter1.fixed_point_iteration_torch import TanhFixedPointLayer
from Chapter1.newton_layer_torch import TanhNewtonLayer
from Chapter1.newton_implicit_layer_torch import TanhNewtonImplicitLayer


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
mnist_train = datasets.MNIST(path, train=True, download=False, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(path, train=False, download=False, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('define net...')
# construct the simple model with fixed point layer

torch.manual_seed(0)
model = nn.Sequential(nn.Flatten(),
                      nn.Linear(784, 100),
                      #TanhFixedPointLayer(100, max_iter=200),
                      TanhNewtonImplicitLayer(100, max_iter=200),
                      nn.Linear(100, 10)
                      ).to(device)
opt = optim.SGD(model.parameters(), lr=1e-1)
is_fixed_point = False


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
