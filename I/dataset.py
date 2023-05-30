import torchvision

from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split

mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


def get_loader(data_path, valid_ratio=0.1, batch_size=128):
    raw_train_set = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=test_transform)
    valid_size = round(len(raw_train_set) * valid_ratio)
    train_set, valid_set = random_split(
        raw_train_set, [len(raw_train_set) - valid_size, valid_size]
    )

    train_loader = DataLoader(train_set, batch_size, True)
    valid_loader = DataLoader(valid_set, batch_size, False)
    test_loader  = DataLoader(test_set, batch_size, False)

    return train_loader, valid_loader, test_loader
