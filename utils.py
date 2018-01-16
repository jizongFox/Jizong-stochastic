'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch,numpy as np,pandas as pd
import torch.nn as nn
import torch.nn.init as init
from torchvision import transforms
import torchvision
from models import *
import torch.backends.cudnn as cudnn
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
term_width =40

TOTAL_BAR_LENGTH = 80.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def transform_dataset(batch_size=256):
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader,testloader, classes


def load_resume(_from):
    print('==> Resuming from '+_from)
    resume_from = _from
    assert os.path.isdir(resume_from), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./'+_from+'/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']+1
    _list = torch.load('./'+_from+'/record.t7')
    train_accuracy_list = _list['train']
    test_accuracy_list = _list['test']
    try:
        learning_rate_list=_list['learning']
    except:
        learning_rate_list={}
    keystoDelete = [x for x in train_accuracy_list.keys() if x > start_epoch]
    for k in keystoDelete:
        train_accuracy_list.pop(k)
        test_accuracy_list.pop(k)
        try:
            learning_rate_list.pop(k)
        except:
            pass
    return (net, best_acc,start_epoch),(train_accuracy_list,test_accuracy_list,learning_rate_list)

def _initilization_(args,use_cuda):
    if args.resume:
        storedNet, trainList=load_resume(_from=args.resume_from)
        (net, best_acc, start_epoch), (train_accuracy_list, test_accuracy_list,learning_rate_list) = storedNet, trainList
        print('best_train_acc:',train_accuracy_list[start_epoch-1],'   best_test_acc:',best_acc,'   ','start_epoch:',start_epoch)
    else:
        print('==> Building model..'+ args.netName)
        listofNet=['VGG','ResNet18','PreActResNet18','GoogLeNet','DenseNet121','ResNeXt29_2x64d','MobileNet','DPN92','ShuffleNetG2','SENet18']
        network_name =args.netName
        assert network_name in listofNet
        if network_name=='VGG':
            net = VGG('VGG19')
        else:
            net = eval(network_name+'()')
        train_accuracy_list = {}
        test_accuracy_list = {}
        learning_rate_list ={}
        best_acc = 0  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    print('resume to '+args.resume_to)

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    return (net, best_acc, start_epoch), (train_accuracy_list, test_accuracy_list,learning_rate_list)


def dump_acc_record(acc,net,use_cuda,epoch,args):

    print('New benchmark, saving..')
    state = {
        'net': net.module if use_cuda else net,
        'acc': acc,
        'epoch': epoch,
    }
    resume_to = args.resume_to
    if not os.path.isdir(resume_to):
        os.mkdir(resume_to)
    try:
        torch.save(state, './'+resume_to+'/ckpt.t7')
    except Exception as e:
        print(e)

def dump_record(train_accuracy_list,test_accuracy_list,learning_rate_list,args):
    # print('saving accuracy history')
    state ={'train':train_accuracy_list,
        'test':test_accuracy_list,
            'learning':learning_rate_list
        }
    if not os.path.isdir(args.resume_to):
        os.mkdir(args.resume_to)
    try:
        torch.save(state, './'+args.resume_to+'/record.t7')
    except Exception as e:
        print(e)
    train = train_accuracy_list
    test = test_accuracy_list
    learning =learning_rate_list
    # epochs = [x for x in list(set(list(train.keys())))]
    # train_score = [train[x] for x in epochs]
    # test_score = [test[x] for x in epochs]

    # plt.figure()
    # plt.plot(epochs, train_score)
    # plt.plot(epochs, test_score)
    # plt.legend(['train', 'test'])
    # plt.show(block=False)
    # try:
    #     plt.savefig('./'+args.resume_to+'/accuracyMap.png')
    # except Exception as e:
    #     print(e)
    # plt.close()
    def _titles_(string, lr, acc):
        string = string + 'lr: ' + str(lr) + ', acc: ' + str(acc) + '\n'
        return string

    epochs = [x for x in list(set(list(train.keys())))]
    train_score = [train[x] for x in epochs]
    test_score = [test[x] for x in epochs]
    learning_score = [learning[x] for x in epochs]
    uniLearning_score = np.unique(learning_score)
    learning = pd.DataFrame(learning, index=[0]).T
    learning.columns = ['lr']
    record_score = {y: max([test[x] for x in learning.lr[learning.lr == y].index]) for y in uniLearning_score}
    string = ''
    for i in sorted(list(uniLearning_score), reverse=True):
        string = _titles_(string, i, record_score[i])
    # print(string)

    ## 最优的点
    maxrecord_index = max(test, key=test.get)
    maxrecord = test[maxrecord_index]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(epochs, train_score)
    ax1.set_ylabel('Accuracy %')
    ax1.plot(epochs, test_score)
    ax1.legend(['train', 'test'], loc='lower left', bbox_to_anchor=(0.15, 0.75), ncol=3, fancybox=True, shadow=True)
    ax1.axvline(maxrecord_index, alpha=0.2, drawstyle='steps-mid', color='r', linewidth=1)
    ax1.axhline(maxrecord, alpha=0.2, drawstyle='steps-mid', color='r', linewidth=1)
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(epochs, np.log10(learning_score), 'g')
    ax2.legend(['learning_rate:\n' + string], loc='lower right')
    ax2.set_ylabel('log(Learning rate)')
    ax2.set_ylim([-6, 3])
    plt.show()
    try:
        plt.savefig('./'+args.resume_to+'/accuracyMap.png')
    except Exception as e:
        print(e)
    plt.close()
