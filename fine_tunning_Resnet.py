#coding=utf-8
import os

## VGG Entropy=> optimized network => stochastic node with low learning rate
cmd1 = '''
python main.py --netName ResNet18 --lr 0.1 --epochs_to_train 150 --resume_to conventional_scheme_ResNet
'''
os.system(cmd1)

cmd2 = '''
python main.py --lr 0.01 --resume --resume_from conventional_scheme_ResNet --resume_to conventional_scheme_ResNet --epochs_to_train 100
'''
os.system(cmd2)

cmd3 = '''
python main.py --lr 0.001 --resume --resume_from conventional_scheme_ResNet --resume_to conventional_scheme_ResNet --epochs_to_train 100
'''
os.system(cmd3)

cmd4 ='''
python main.py --lr 0.0001 --resume --resume_from conventional_scheme_ResNet --resume_to fine-tunning_ResNet --loss_function stochastic --normalize --epochs_to_train 10000
'''
os.system(cmd4)

