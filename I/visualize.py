import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import pickle
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

visualize_transform = transforms.Compose([
    transforms.ToTensor()
])

def mixup(images, labels, ratio=torch.tensor(0.6)):
    images_a = images
    labels_a = labels
    images_b = torch.cat((images[1:], images[0].unsqueeze(0)))
    labels_b = torch.cat((labels[1:], labels[0].unsqueeze(0)))
    ratio = ratio.to(images_a.device)

    images = images_a*ratio.view(-1, 1, 1, 1) + images_b*(1-ratio.view(-1, 1, 1, 1))
    labels_a = F.one_hot(labels_a, num_classes=100)
    labels_b = F.one_hot(labels_b, num_classes=100)
    labels = labels_a*ratio.view(-1, 1) + labels_b*(1-ratio.view(-1, 1))

    return images, labels

def cutout(images, labels):
    batch_size = labels.size(0)
    images = images.clone()
    offset = torch.tensor(4.0).view(-1, 1)
    center = torch.randint(0, 32, (batch_size, 2))
    lower = torch.round(torch.clip(center - offset, min=0)).to(torch.long)
    upper = torch.round(torch.clip(center + offset, max=32)).to(torch.long)

    for i in range(batch_size):
        images[i, :, lower[i, 0]:upper[i, 0], lower[i, 1]:upper[i, 1]] = 0

    return images, labels

def cutmix(images, labels, ratio=torch.tensor(0.6)):
    batch_size = labels.size(0)
    images = images.clone()
    images_a = images
    labels_a = labels
    images_b = torch.cat((images[1:], images[0].unsqueeze(0)))
    labels_b = torch.cat((labels[1:], labels[0].unsqueeze(0)))
    ratio = ratio.to(images_a.device).unsqueeze(0).repeat(3, 1)

    offset = torch.sqrt(1 - ratio) * 16
    offset = offset.view(-1, 1)
    center = torch.randint(0, 32, (batch_size, 2)).to(offset.device)
    lower = torch.round(torch.clip(center - offset, min=0)).to(torch.long)
    upper = torch.round(torch.clip(center + offset, max=32)).to(torch.long)

    for i in range(batch_size):
        images[i, :, lower[i, 0]:upper[i, 0], lower[i, 1]:upper[i, 1]] = images_b[i, :, lower[i, 0]:upper[i, 0], lower[i, 1]:upper[i, 1]]
        ratio[i] = 1 - (upper[i, 0] - lower[i, 0]) * (upper[i, 1] - lower[i, 1]) / 1024

    labels_a = F.one_hot(labels_a, num_classes=100)
    labels_b = F.one_hot(labels_b, num_classes=100)
    labels = labels_a * ratio.view(-1, 1) + labels_b * (1 - ratio.view(-1, 1))

    return images, labels

if __name__ == "__main__":
    train_set = torchvision.datasets.CIFAR100("/home/jgc22/NN/I/data/", True, transform=visualize_transform)
    train_loader = DataLoader(train_set, 3, True)

    images, labels = next(iter(train_loader))
    mixup_images, mixup_labels = mixup(images, labels, torch.tensor(0.6))
    cutout_images, cutout_labels = cutout(images, labels)
    cutmix_images, cutmix_labels = cutmix(images, labels, torch.tensor(0.6))
    with open("/home/jgc22/NN/I/data/cifar-100-python/meta", "rb") as f:
        cifar100_meta = pickle.load(f, encoding="bytes")

    label_names = cifar100_meta[b"fine_label_names"]
    fig, axs = plt.subplots(3, 4, figsize=(10, 6))

    for i in range(3):
        axs[i, 0].imshow(images[i].permute(1, 2, 0))
        axs[i, 0].set_title(f"{label_names[labels[i]]}")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(mixup_images[i].permute(1, 2, 0))
        axs[i, 1].set_title(f"Mixup: {label_names[labels[i]]} + {label_names[labels[(i+1)%3]]}")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(cutout_images[i].permute(1, 2, 0))
        axs[i, 2].set_title(f"Cutout: {label_names[labels[i]]}")
        axs[i, 2].axis("off")

        axs[i, 3].imshow(cutmix_images[i].permute(1, 2, 0))
        axs[i, 3].set_title(f"Cutmix: {label_names[labels[i]]} + {label_names[labels[(i+1)%3]]}")
        axs[i, 3].axis("off")

    plt.savefig('./visualize.jpg')