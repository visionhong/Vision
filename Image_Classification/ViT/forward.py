import numpy as np
from PIL import Image
import torch
import torch.nn.functional
import cv2

k = 10

imagenet_labels = dict(enumerate(open("classes.txt")))

model = torch.load("ViT.pth")
model.eval()


img = (np.array(Image.open("cat.jpg"))/128) - 1  # -1~1
img = cv2.resize(img, (384, 384))
inp = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
logits = model(inp)
probs = torch.nn.functional.softmax(logits, dim=1)

top_probs, top_idxs = probs[0].topk(k)

for i, (idx_, prob_) in enumerate(zip(top_idxs, top_probs)):
    idx = idx_.item()
    prob = prob_.item()
    cls = imagenet_labels[idx].strip()
    print(f"{i}: {cls:<45} --- {prob:.4f}")
