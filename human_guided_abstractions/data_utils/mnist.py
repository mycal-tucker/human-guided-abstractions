import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from human_guided_abstractions.data_utils.mini_dataset import MiniByLabel

"""
Helper methods for loading data. Because the data and loss function and network architectures are all linked,
we return lots of info. The general format of returned values from all methods is:

1) train data
2) test data
3) number of output neurons the classifier should have
4) the training criterion for the classifier
5) the (optional) type the labels should be cast to.
"""


def _raw_fashion_mnist():
    transform = transforms.Compose(
        [transforms.ToTensor()])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                 download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform)
    return trainset, testset


def get_fashion_classification(batch_size, trainset_size=None, choosing_idx=None):
    trainset, testset = _raw_fashion_mnist()
    trainset = FashionMNISTTernary(trainset)
    testset = FashionMNISTTernary(testset)
    if trainset_size is not None:
        choosing_idx = 1 if choosing_idx is None else choosing_idx
        num_labels = 10 if choosing_idx == 1 else 3
        trainset = MiniByLabel(trainset, trainset_size, num_labels, choosing_idx)
        trainset = MiniByLabel(trainset, trainset_size, num_labels, choosing_idx)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2048,
                                             shuffle=False)
    original_trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                    download=True, transform=transforms.ToTensor())
    return trainloader, testloader, 10, 3, nn.CrossEntropyLoss(), None, original_trainset

class FashionMNISTTernary(torch.utils.data.Dataset):
    def __init__(self, raw_dataset):
        self.raw_dataset = raw_dataset

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        img, fine_label = self.raw_dataset.__getitem__(idx)
        crude_label = None
        # Normal hierarchy
        if fine_label in [0, 2, 4, 6]:  # Top: Tshirt/top, Pullover, Coat, Shirt
            crude_label = 0
        elif fine_label in [1, 3, 8]:  # Trouser, Dress, Bag
            crude_label = 1
        elif fine_label in [5, 7, 9]:  # Sandal, Sneaker, Ankle boot
            crude_label = 2
        crude_label = torch.tensor(crude_label)
        return img, crude_label, fine_label, idx
