import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch.nn.modules.loss import _assert_no_grad
from torch.autograd import Variable
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import time

class zeroneLoss(_Loss):
    def __init__(self, size_average=True, reduce=True, normalize=False,correct_reward=0.5, incorrect_penality=0):
        super(zeroneLoss, self).__init__(size_average)
        self.reduce = reduce
        self.normalize =normalize
        self.correct_reward=correct_reward
        self.incorrect_pentality=incorrect_penality
    def forward(self, input, target):
        _assert_no_grad(target)  #
        prob = F.softmax(input,dim=1)
        m = torch.distributions.Categorical(prob)
        sel2 = m.sample().type(torch.LongTensor)  # replacement = True
        rewards = ((target.view(sel2.size()) == sel2) > 0).type(torch.FloatTensor)
        rewards [rewards ==1]=self.correct_reward
        rewards [rewards ==0]= self.incorrect_pentality
        if self.norm==True:
            self.meanreward = rewards.mean()
            rewards = (rewards - self.meanreward) / (rewards.std() + float(np.finfo(np.float32).eps))
        loss = - m.log_prob(sel2) * rewards
        return loss.mean()

class EntropyLoss_test(_Loss):
    def __init__(self,beta):
        super(EntropyLoss_test,self).__init__()
        self.beta = beta
    def forward(self, netoutput, target):
        probs  = F.softmax(netoutput,dim=1)
        likely_prob = probs.gather(1,target.view(-1,1))
        crossEntropy = -likely_prob.log().mean()
        self_entropy = -(probs*probs.log()).sum(1).mean()

        return crossEntropy - self.beta*self_entropy

def parser_loss_function(args):
    chooseList=['cross_entropy','cross_entropy_entropy','stochastic']
    assert args.loss_function in chooseList

    if args.loss_function == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif args.loss_function=='stochastic':
        return zeroneLoss(normalize=args.normalize,correct_reward=args.correct_reward,incorrect_pentality= args.incorrect_penalty)
    elif args.loss_function=='cross_entropy_entropy':
        pass
    # TODO