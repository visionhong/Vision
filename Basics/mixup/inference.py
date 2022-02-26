import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import data_loader, plot_examples, restore
from mobilenet import MobileNet
import pdb


classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
trainloader, testloader = data_loader(16)
net = MobileNet().cuda()

if os.path.isdir("checkpoint"):
    path = "checkpoint/ckpt.t70_20210324"
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint)


def visualize_model(model):
    was_training = model.training
    model.eval()
    fig = plt.figure()
    images = []
    title = []
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(testloader):
            inputs = inputs.cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                inp = restore(inputs[j].cpu())
                images.append(inp)
                title.append(classes[preds[j]])

            plot_examples(images,title)
            break


if __name__ == '__main__':
    visualize_model(net)