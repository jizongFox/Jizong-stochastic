import torch,matplotlib.pyplot as plt
try:
    accuarcy_list = torch.load('record.t7')
except:
    accuarcy_list = torch.load('checkpoint_sto/record.t7')
train =accuarcy_list['train']
test = accuarcy_list['test']
epochs =[x for x in list(set(list(train.keys())))]
train_score = [train[x] for x in epochs]
test_score = [test[x] for x in epochs]

plt.figure()
plt.plot(epochs,train_score)
plt.plot(epochs,test_score)
plt.legend(['train','test'])
plt.show()