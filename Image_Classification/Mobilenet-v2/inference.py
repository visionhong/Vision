from train_test import *
from MobilenetV2 import mobilenet_v2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image



class Archive(Dataset):
    def __init__(self, path, transform=None):
        img_name = [f for f in os.listdir(path)]
        self.imgList = [os.path.join(path, i) for i in img_name]
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        image = Image.open(self.imgList[idx]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, inputs in enumerate(pred_loader):
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                print(preds[j])
                ax.set_title('predicted: {}'.format(classes[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_path = 'E:\\Computer Vision\\data\\archive\\seg_train\\'
    pred_path = 'E:\\Computer Vision\\data\\archive\\seg_pred\\'
    classes = sorted(os.listdir(train_path))

    pred_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    pred_dataset = Archive(pred_path, transform=pred_transform)
    pred_loader = DataLoader(pred_dataset, batch_size=64, shuffle=True, num_workers=6, pin_memory=True)

    model = mobilenet_v2(False)
    model.classifier = nn.Linear(model.classifier.in_features, len(classes)).to(device)  # 1000 -> 6

    checkpoint = torch.load('checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])

    model = model.to(device)
    visualize_model(model)