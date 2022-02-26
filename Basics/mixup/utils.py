import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pdb

def data_loader(batch_size):
    # Data
    print('==> Preparing data..')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                                transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = datasets.CIFAR10(root='./data', train=False, download=False,
                                transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)
    return trainloader, testloader

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''
    mixup은 training에서만 사용한다.
    train에서는 정확도가 낮게나오지만(이미지는 두개지만 정답은 max값 1개로 취급하기 때문) test에서는 높게 나온다.
    이것도 label smoothing과 같은 선상에서 이해하면 좋을 것 같다.
    label smoothing은 incoding된 hard한 정답이 일반화에 방해가 되어 overfitting을 일으킨다는 가정에서 나온 기법인데
    mixup 또한 두개의 의미지를 섞어서 일반화를 위해 사용되는 것 같다.
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()  # batch_size만큼의 random index 1차원 벡터\

    else:
        index = torch.randperm(batch_size)
    # pdb.set_trace()
    mixed_x = lam * x + (1 - lam) * x[index, :]  # 원본 batch image와 섞인 batch image들을 mix
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1-lam) * criterion(pred, y_b) # cross entropy에 각각 합성된 비율(lambda)값을 곱함


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def restore(inp, title=None):
    """Imshow for Tensor."""
    # pdb.set_trace()
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def plot_examples(images, title=None):
    fig = plt.figure(figsize=(10, 10))
    columns = 4
    rows = 4

    for i in range(1, len(images)+1):
        img = images[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        if title is not None:
            plt.title(title[i-1])
        plt.tight_layout()
    plt.show()

