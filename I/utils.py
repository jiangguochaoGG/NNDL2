import torch
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

def mixup(images, labels, ratio=None):
    batch_size = labels.size(0)
    images_a = images
    labels_a = labels
    shuffle_index = torch.randperm(batch_size, dtype=torch.long)
    images_b = images[shuffle_index]
    labels_b = labels[shuffle_index]

    if ratio is None:
        ratio = Uniform(low=0, high=1).sample((batch_size, )).to(images_a.device)
    else:
        ratio = ratio.to(images_a.device)

    images = images_a*ratio.view(-1, 1, 1, 1) + images_b*(1-ratio.view(-1, 1, 1, 1))
    labels_a = F.one_hot(labels_a, num_classes=100)
    labels_b = F.one_hot(labels_b, num_classes=100)
    labels = labels_a*ratio.view(-1, 1) + labels_b*(1-ratio.view(-1, 1))

    return images, labels

def cutout(images, labels):
    batch_size = labels.size(0)
    offset = torch.tensor(4.0).view(-1, 1)
    center = torch.randint(0, 32, (batch_size, 2))
    lower = torch.round(torch.clip(center - offset, min=0)).to(torch.long)
    upper = torch.round(torch.clip(center + offset, max=32)).to(torch.long)

    for i in range(batch_size):
        images[i, :, lower[i, 0]:upper[i, 0], lower[i, 1]:upper[i, 1]] = 0

    return images, labels

def cutmix(images, labels):
    batch_size = labels.size(0)
    images_a = images
    labels_a = labels
    shuffle_index = torch.randperm(batch_size, dtype=torch.long)
    images_b = images[shuffle_index]
    labels_b = labels[shuffle_index]

    ratio = Uniform(low=0, high=1).sample((batch_size, )).to(images_a.device)

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
