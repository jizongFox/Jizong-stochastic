import torch,matplotlib.pyplot as plt
import numpy as np,pandas as pd

def _titles_(string,lr, acc):
    string = string +'lr: '+str(lr)+', acc: '+str(acc)+'\n'
    return string
try:
    accuarcy_list = torch.load('record.t7')
except:
    accuarcy_list = torch.load('checkpoint_sto/record.t7')
train =accuarcy_list['train']
test = accuarcy_list['test']
learning = accuarcy_list['learning']
epochs =[x for x in list(set(list(train.keys())))]
train_score = [train[x] for x in epochs]
test_score = [test[x] for x in epochs]
learning_score = [learning[x] for x in epochs]
uniLearning_score = np.unique(learning_score)
learning = pd.DataFrame(learning,index=[0]).T
learning.columns=['lr']
record_score={y: max([test[x] for x in learning.lr[learning.lr==y].index]) for y in uniLearning_score}
string =''
for i in sorted(list(uniLearning_score),reverse=True):
    string=_titles_(string,i, record_score[i])
# print (string)

## 最优的点
maxrecord_index = max(test, key=test.get)
maxrecord = test[maxrecord_index]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(epochs,train_score)
ax1.set_ylabel('Accuracy %')
ax1.plot(epochs,test_score)
ax1.legend(['train','test'],loc='lower left', bbox_to_anchor=(0.15,0.75),ncol=3,fancybox=True,shadow=True)
# print(np.unique(np.log10(learning_score)))
ax1.axvline(maxrecord_index,alpha=0.2,drawstyle='steps-mid',color='r',linewidth=1)
ax1.axhline(maxrecord,alpha=0.2,drawstyle='steps-mid',color='r',linewidth=1)
ax2 = ax1.twinx()  # this is the important function
ax2.plot(epochs,np.log10(learning_score),'g')
ax2.legend(['learning_rate:\n'+string],loc='lower right')
ax2.set_ylabel('log(Learning rate)')
ax2.set_ylim([-6,3])
plt.show()