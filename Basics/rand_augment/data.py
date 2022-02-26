import torchvision
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import SubsetRandomSampler, Sampler, DataLoader
from augmentations import RandAugment


def get_dataloaders(batch, dataroot, split=0.15, split_idx=0, rand_aug=False):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    if rand_aug:
        transform_train.transforms.insert(0, RandAugment(n=1, m=2))  # Rand Augment

    total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)

    if split > 0.0:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=split, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)

    trainloader = DataLoader(total_trainset, batch_size=batch, shuffle=True if train_sampler is None else False,
                             num_workers=2, pin_memory=True, sampler=train_sampler, drop_last=True)
    validloader = DataLoader(total_trainset, batch_size=batch, shuffle=False,
                             num_workers=2, pin_memory=True, sampler=valid_sampler, drop_last=False)
    testloader = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=2,
                            pin_memory=True, drop_last=False)
    return train_sampler, trainloader, validloader, testloader


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)