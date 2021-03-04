# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import pandas as pd
import csv

from models import *
from models.vit import ViT
from utils import progress_bar, Subset, dump_stats
from torch.utils.tensorboard import SummaryWriter

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='add image augumentations')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', type=int, default='12')
parser.add_argument('--n_epochs', type=int, default='15')
parser.add_argument('--patch', default='4', type=int)
parser.add_argument('--cos', action='store_true', help='Train with cosine annealing scheduling')
parser.add_argument('--use_attention_entropy', required=True, type=int, help='whether to use entropy or not') # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--max_batches_ae', type=int, default=50, help='Number of batches to use attention entropy.')
parser.add_argument('--ae_weight', type=float, default=0.1, help='Coefficient for attention entropy penalty')
parser.add_argument('--random_seed', type=int, default=123, help='Random Seed for experiments.')
parser.add_argument('--run', type=int, default=0, help='run id.')
parser.add_argument('--proportion', type=float, default=0.9, help='Proportion of Dataset to use for training')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for dataloader')

args = parser.parse_args()
print(args)

import numpy as np
np.random.seed(args.random_seed)
torch.random.manual_seed(args.random_seed)
torch.set_deterministic(True)

ENTROPY_WEIGHT = args.ae_weight

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_loss = float('inf')
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Tensorboard writer
tb_writer = SummaryWriter(f"./log/{args.net}_ae_{args.use_attention_entropy}_run_{args.run}/")

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainset = Subset(trainset, args.proportion)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='res18':
    net = ResNet18()
elif args.net=='vgg':
    net = VGG('VGG19')
elif args.net=='res34':
    net = ResNet34()
elif args.net=='res50':
    net = ResNet50()
elif args.net=='res101':
    net = ResNet101()
elif args.net=="vit":
    # ViT for cifar10
    net = ViT(
    image_size = 32,
    patch_size = args.patch,
    num_classes = 10,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net == "tnt":
    net = TNT(
        image_size = 32,       # size of image
        patch_dim = 512,        # dimension of patch token
        pixel_dim = 24,         # dimension of pixel token
        patch_size = 4,        # patch size
        pixel_size = 2,         # pixel size
        depth = 6,              # depth
        num_classes = 10,     # output number of classes
        attn_dropout = 0.1,     # attention dropout
        ff_dropout = 0.1        # feedforward dropout
    )

net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net) # make parallel
#     cudnn.benchmark = True


# Loss is CE
criterion = nn.CrossEntropyLoss()

# reduce LR on Plateau
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)

##### Training
step = 0
def train(epoch):
    global ENTROPY_WEIGHT, step
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, ent = net(inputs, return_ent=True)
        loss_ce = criterion(outputs, targets)
        loss_ae = ENTROPY_WEIGHT * ent        
        if step < args.max_batches_ae and args.use_attention_entropy == 1:
            loss = loss_ae + loss_ce
        else:
            loss = loss_ce

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        tb_writer.add_scalar("Accuracy/train", 100.*correct/total, step)
        tb_writer.add_scalar("LossCE/train", loss_ce.item(), step)
        tb_writer.add_scalar("LossAE/train", loss_ae.item(), step)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        step += 1

    return train_loss/(batch_idx+1)

##### Validation
import time
def test(epoch):
    global best_acc, best_loss, step
    net.eval()
    test_loss = 0
    total_loss_ae = 0
    correct = 0
    total = 0
    num_batches = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, ent = net(inputs, return_ent=True)
            loss = criterion(outputs, targets)
            total_loss_ae += ENTROPY_WEIGHT * ent.item()

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            num_batches += 1

    tb_writer.add_scalar("Accuracy/val", 100.*correct/total, step)
    tb_writer.add_scalar("LossCE/val", test_loss / num_batches, step)
    tb_writer.add_scalar("LossAE/val", total_loss_ae / num_batches, step)

    # Update scheduler
    if not args.cos:
        scheduler.step(test_loss)
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc
        best_loss = test_loss / num_batches
        
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []
for epoch in range(start_epoch, args.n_epochs):
    # with torch.autograd.detect_anomaly():
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    if args.cos:
        scheduler.step(epoch-1)
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # write as csv for analysis
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print(list_loss)

stats = {
    "name": f"{args.net}_ae_{args.use_attention_entropy}_run_{args.run}",
    "best_test_acc": best_acc,
    "best_test_loss": best_loss,
    "use_attention_entropy": args.use_attention_entropy,
    "run": args.run,
    "max_batches_ae": args.max_batches_ae,
    "random_seed": args.random_seed,
    "train_proportion": args.proportion,
    "train_size": len(trainset)
}

dump_stats(stats)