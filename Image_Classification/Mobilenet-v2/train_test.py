'''
Download Data : https://www.kaggle.com/puneet6060/intel-image-classification
'''

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from MobilenetV2 import mobilenet_v2
import tqdm


def train(epoch):
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    for index, (inputs, targets) in enumerate(tqdm.tqdm(train_loader, desc='TRAIN')):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    print(f'Train epoch: {epoch+1} loss: {train_loss / len(train_loader):.4f} | Acc: {correct / total * 100:.4f}')


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for index, (inputs, targets) in enumerate(tqdm.tqdm(test_loader, desc='TEST')):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        print(f'Test epoch: {epoch+1} loss: {test_loss / len(test_loader):.4f} | Acc: {correct / total * 100:.4f}')

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_path = 'E:\\Computer Vision\\data\\archive\\seg_train\\'
    test_path = 'E:\\Computer Vision\\data\\archive\\seg_test\\'

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_path, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

    classes = os.listdir(train_path)

    model = mobilenet_v2(True)  # take pretrained weights
    model.classifier = nn.Linear(model.classifier.in_features, len(classes)).to(device)  # change classifier
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_acc = 0

    # Fine tuning
    for epoch in range(10):
        train(epoch)
        test(epoch)
