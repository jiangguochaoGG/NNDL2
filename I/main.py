import os
import random
import numpy as np
import torch
import argparse
import pickle

from tqdm import tqdm
from model import resnet18
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from dataset import get_loader
from torch.utils.tensorboard import SummaryWriter
from utils import mixup, cutout, cutmix

def set_seed(seed):
    # 设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=4396, type=int)
    parser.add_argument(
        '--data_path',
        default='/home/jgc22/NN/I/data/',
        type=str
    )
    parser.add_argument(
        '--output_path',
        default='/home/jgc22/NN/I/output/',
        type=str
    )
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--wd', default=5e-4, type=float)
    parser.add_argument('--batch_size', default=512, type=int)

    parser.add_argument('--mixup', default=False, type=bool)
    parser.add_argument('--cutout', default=False, type=bool)
    parser.add_argument('--cutmix', default=False, type=bool)

    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return args

def train(args, model, optimizer, scheduler, criterion, tb_writer):
    runs = []
    epoch = args.epoch
    train_loader, valid_loader, test_loader = get_loader(args.data_path, batch_size=args.batch_size)
    best_acc, best_epoch = 0, 0

    for i in range(1, epoch + 1):
        print(f"Epoch {i}:")
        tb_writer.add_scalar(tag="epoch", scalar_value=i, global_step=i)
        total_loss = 0.0
        total_count = 0
        total_acc = 0
        model.train()

        train_tqdm = tqdm(train_loader)
        for _, batch in enumerate(train_tqdm):
            images, labels = batch
            images = images.to(args.device)
            labels = labels.to(args.device)
            if args.mixup:
                images, labels = mixup(images, labels)
            elif args.cutout:
                images, labels = cutout(images, labels)
            elif args.cutmix:
                images, labels = cutmix(images, labels)
            optimizer.zero_grad()

            preds = model(images)
            loss = criterion(preds, labels)

            loss.backward()
            total_loss += loss.item()
            total_count += labels.size(0)
            if len(labels.shape) == 2:
                total_acc += torch.sum(preds.argmax(dim=1) == labels.argmax(dim=1)).item()
            else:
                total_acc += torch.sum(preds.argmax(dim=1) == labels).item()

            train_tqdm.set_description("loss: {:.5f}".format(round(loss.item(), 5)))
            optimizer.step()

        print("Train loss: {:.5f} Train Acc: {:.5f}".format(total_loss / total_count, total_acc / total_count))
        tb_writer.add_scalar(tag="loss/train", scalar_value=total_loss / total_count, global_step=i)
        tb_writer.add_scalar(tag="acc/train", scalar_value=total_acc / total_count, global_step=i)

        valid_loss, valid_acc = valid(args, model, valid_loader, criterion)
        print("Valid loss: {:.5f} Valid Acc: {:.5f}".format(valid_loss, valid_acc))
        tb_writer.add_scalar(tag="loss/val", scalar_value=valid_loss, global_step=i)
        tb_writer.add_scalar(tag="acc/val", scalar_value=valid_acc, global_step=i)
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = i
            with open(args.output_path + "best.pt", 'wb') as f:
                torch.save(model, f)
        runs.append((total_loss / total_count, total_acc / total_count, valid_loss, valid_acc))
        if scheduler is not None:
            scheduler.step()

    with open(args.output_path + "best.pt", 'rb') as f:
        model = torch.load(f).to(args.device)

    _, test_acc = valid(args, model, test_loader, criterion)
    print("Best Epoch: {} Test Acc: {}".format(best_epoch, test_acc))
    tb_writer.add_scalar(tag="best_epoch", scalar_value=best_epoch)
    tb_writer.add_scalar(tag="acc/test", scalar_value=test_acc)

    with open(args.output_path + 'runs.pickle', 'wb') as f:
        pickle.dump(runs, f)
    with open(args.output_path + 'test_acc.pickle', 'wb') as f:
        pickle.dump(test_acc, f)

def valid(args, model, valid_loader, criterion):
    with torch.no_grad():
        model.eval()
        all_pred, all_label = [], []
        total_loss = 0.0
        valid_tqdm = tqdm(valid_loader, desc="Validating...")
        for images, labels in valid_tqdm:
            images = images.to(args.device)
            labels = labels.to(args.device)
            preds = model(images)
            total_loss += criterion(preds, labels).item()
            all_pred.append(preds)
            all_label.append(labels)

        all_pred = torch.cat(all_pred)
        all_label = torch.cat(all_label)
        acc = torch.sum(
            all_pred.argmax(dim=1) == all_label
        ).item() / all_label.size(0)

    return total_loss / all_label.size(0), acc



if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    tb_writer = SummaryWriter("./runs/")

    model = resnet18().to(args.device)
    optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    criterion = CrossEntropyLoss()

    train(args, model, optimizer, scheduler, criterion, tb_writer)
