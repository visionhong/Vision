from utils import data_loader, plot_examples, restore
import torch
from utils import mixup_data
import pdb


def visualization():
    trainloader, _ = data_loader(16)
    images = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)

        for j in range(inputs.size(0)):
            inp = restore(inputs[j].cpu())
            images.append(inp)

        plot_examples(images)
        break

if __name__ == '__main__':
    visualization()