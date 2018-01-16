#coding=utf-8
'''Train CIFAR10 with PyTorch.'''
import torch.nn as nn
import torch.optim as optim,torch
import os
import argparse
from utils import progress_bar, transform_dataset,_initilization_,dump_acc_record,dump_record
from loss_Functions import  parser_loss_function
from torch.autograd import Variable

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    train_accuracy_list[epoch] = 100. * correct / total
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    test_accuracy_list[epoch] = 100. * correct / total
    acc = 100. * correct / total
    if acc > best_acc:
        dump_acc_record(acc, net, use_cuda, epoch, args)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume_from',default='checkpoint',type=str,help='resume from which checkpoint')
    parser.add_argument('--resume_to',default='checkpoint',type=str,help='resume to which directory')
    parser.add_argument('--netName',default='VGG',type=str,help='choose the net')
    parser.add_argument('--loss_function',default='cross_entropy',help='change to different loss functions')
    parser.add_argument('--correct_reward', default=1, type=float, help='correct_reward')
    parser.add_argument('--incorrect_penalty', default=0, type=float, help='incorrect_penalty')
    parser.add_argument('--normalize', action='store_true', help='normalize the rewards')
    parser.add_argument('--epochs_to_train',default=300,type=int)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    trainloader, testloader, _ = transform_dataset(batch_size=400)

    # Model
    storedNet, trainList = _initilization_(args,use_cuda)
    (net, best_acc, start_epoch), (train_accuracy_list, test_accuracy_list,learning_rate_list) = storedNet, trainList

    criterion = parser_loss_function(args=args)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Training
    for epoch in range(start_epoch, start_epoch+args.epochs_to_train):
        train(epoch)
        test(epoch)
        if epoch %3 ==0:
            dump_record(train_accuracy_list,test_accuracy_list,learning_rate_list,args)
