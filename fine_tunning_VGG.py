#coding=utf-8
import os

# VGG Entropy=> optimized network => stochastic node with low learning rate
cmd1 = '''
python main.py --lr 0.1 --resume_to conventional_scheme_VGG  --epochs_to_train 100
'''
os.system(cmd1)

cmd2 = '''
python main.py --lr 0.01 --resume --resume_from conventional_scheme_VGG --resume_to conventional_scheme_VGG --epochs_to_train 100
'''
os.system(cmd2)

cmd3 = '''
python main.py --lr 0.001 --resume --resume_from conventional_scheme_VGG --resume_to conventional_scheme_VGG --epochs_to_train 100
'''
os.system(cmd3)

cmd4 ='''
python main.py --lr 0.0001 --resume --resume_from conventional_scheme_VGG --resume_to fine-tunning_VGG --loss_function stochastic --normalize --epochs_to_train 5000
'''
os.system(cmd4)

