import torch
import pandas as pd
import numpy as np
import config
from tqdm import tqdm
import warnings
import torch.nn.functional as F
import pdb

def make_prediction(model, loader, output_csv='submission.csv'):
    preds = []
    filenames = []
    model.eval()

    for x, y, files in tqdm(loader):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            predictions = model(x)
            # Convert MSE floats to integer predictions
            predictions[predictions < 0.5] = 0
            predictions[(predictions >= 0.5) & (predictions < 1.5)] = 1
            predictions[(predictions >= 1.5) & (predictions < 2.5)] = 2
            predictions[(predictions >= 2.5) & (predictions < 3.5)] = 3
            predictions[(predictions >= 3.5) & (predictions < 10000000)] = 4

            predictions = predictions.long().squeeze(1)
            preds.append(predictions.cpu().numpy())
            filenames += files

    df = pd.DataFrame({'image':filenames, 'level':np.concatenate(preds, axis=0)})
    df.to_csv(output_csv, index=False)
    model.train()
    print('Done with predictions')


def check_accuracy(model, loader, device='cuda'):
    model.eval()
    all_preds ,all_labels = [], []
    num_correct = 0
    num_samples = 0

    for x, y, filename in tqdm(loader):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            predictions = model(x)

        predictions[predictions < 0.5] = 0
        predictions[(predictions >= 0.5) & (predictions < 1.5)] = 1
        predictions[(predictions >= 1.5) & (predictions < 2.5)] = 2
        predictions[(predictions >= 2.5) & (predictions < 3.5)] = 3
        predictions[(predictions >= 3.5) & (predictions < 100)] = 4
        pdb.set_trace()
        predictions = predictions.long().view(-1)
        y = y.view(-1)

        num_correct += (predictions == y).sum()
        num_samples += predictions.shape[0]

        # add to lists
        all_preds.append(predictions.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )
        model.train()
        return np.concatenate(all_preds, axis=0, dtype=np.int64), np.concatenate(
            all_labels, axis=0, dtype=np.int64
        )

def save_checkpoint(state, file_name='my_checkpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state, file_name)

def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_csv_for_blend():
    pass











