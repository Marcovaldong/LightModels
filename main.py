import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from models.squeezenet import SqueezeNet
from models.mobilenet import MobileNet
from models.mobilenetv2_bak import MobileNetV2
from models.shufflenet import shufflenet
from models.senet import SENet18
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser('Options for training SqueezeNet in pytorch')
parser.add_argument('--model', type=str, default='SqueezeNet', help='name of model we use to train')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epoch', type=int, default=55, help='number of epoch to train')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='percentage of past parameters to store')
parser.add_argument('--use_cuda', action='store_true', default=True, help='use cuda for training')
parser.add_argument('--log_schedule', type=int, default=10, help='number of batch size to save snapshot after')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--pretrained', type=int, default=None, help='use a pretrained model')
parser.add_argument('--want_to_test', type=bool, default=False, help='make true if you just want to test')
parser.add_argument('--epoch_55', action='store_true', help='would you like to use 55 epoch learning rule')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))
                     ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False,
                     transform=transforms.Compose([
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))
                     ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

if args.model == 'SqueezeNet':
    net = SqueezeNet()
elif args.model == 'MobileNet':
    net = MobileNet(num_classes=10)
elif args.model == 'ShuffleNet':
    net = shufflenet(groups=2)
elif args.model == 'MobileNetv2':
    net = MobileNetV2()
elif args.model == 'SENet':
    net = SENet18()


if args.pretrained is not None:
    print('Loading pretrained weights...')
    net.load_state_dict(torch.load(args.pretrained))

if args.cuda:
    net.cuda()

# print(net)

# create optimizer
avg_loss = list()
best_accuracy = 0.0
fig1, ax1 = plt.subplots()

optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

def adjustlrwd(params):
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = params['learning_rate']
        param_group['weight_decay'] = params['weight_decay']

def train(epoch):
    # if args.epoch_55:
    #     params = paramsforepoch(epoch)
    #     print('Configuring optimizer with lr={:.5f} and weight_decay={:.4f}'.format(params['learning_rate'], params['weight_decay']))
    #     adjustlrwd(params)

    global avg_loss
    correct = 0
    net.train()
    for idx, (data, label) in enumerate(train_loader):
        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)

        optimizer.zero_grad()
        scores = net.forward(data)
        # print('scores', scores.data)
        # print('label', label.data)
        loss = criterion(scores, label) # F.nll_loss(scores, label)

        # compute the accuracy
        pred = scores.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(label.data).cpu().sum()

        avg_loss.append(loss.data[0])
        loss.backward()
        optimizer.step()

        if idx % args.log_schedule == 0:
            print('Epoch [{}/{}] Iter [{}/{}]\tLoss {:.4f}'.format(
                epoch, args.epoch, (idx+1) * len(data), len(train_loader.dataset),
                loss.data[0]))
            # plot the loss, it should go down exponentially at some point
            ax1.plot(avg_loss)
            fig1.savefig('./outputs/{}_loss.jpg'.format(args.model))

    train_acc = correct / float(len(train_loader.dataset))
    print('training accuracy ({:.2f}%)'.format(100 * train_acc))
    return train_acc * 100.0

def val(epoch):
    global best_accuracy
    correct = 0
    net.eval()
    idx = 0
    for idx, (data, label) in enumerate(test_loader):
        # if idx == 73:
        #     break
        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)

        # do the forward pass
        score = net.forward(data)
        pred = score.data.max(1)[1] # got the indices of the maximum, match them
        correct += pred.eq(label.data).cpu().sum()

    print('predicted {} out of {} at epoch {}'.format(correct, len(test_loader.dataset), epoch))
    val_acc = float(correct) / len(test_loader.dataset) * 100
    print('val accuracy = {:.2f}%...'.format(val_acc))

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(net.state_dict(), './checkpoint/{}.pth'.format(args.model))
    return val_acc

def test():
    net.load_state_dict('./checkpoint/{}.pth'.format(args.model))
    net.eval()

    test_correct = 0
    total_examples = 0
    acc = 0.0
    for idx, (data, label) in enumerate(test_loader):
        if idx < 73:
            continue
        total_examples += len(label)
        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)

        scores = net(data)
        pred = scores.data.max(1)[1]
        test_correct += pred.eq(label.data).cpu().max()
    print('Predicted {} out of {} correctly'.format(test_correct, total_examples))
    return 100.0 * test_correct / (float(total_examples))

if __name__ == '__main__':
    start = time.time()
    if not args.want_to_test:
        fig2, ax2 = plt.subplots()
        train_acc, val_acc = list(), list()
        for i in range(1, args.epoch+1):
            train_acc.append(train(i))
            thistime = time.time()
            print("the model has been training for {:.2f} seconds".format(thistime - start))
            val_acc.append(val(epoch=i))
            ax2.plot(train_acc, 'g')
            ax2.plot(val_acc, 'b')
            fig2.savefig('./outputs/{}_train_val_acc.jpg'.format(args.model))
    else:
        test_acc = test()
        print('Test accuracy on CIFAR-10 is {.2f}%'.format(test_acc))