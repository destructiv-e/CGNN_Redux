import sys
import os
import copy
import json
import datetime
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

opt = dict()

opt['dataset'] = 'dataset/out_cora'
opt['hidden_dim'] = 16
opt['input_dropout'] = 0.5
opt['dropout'] = 0
opt['optimizer'] = 'adam'
opt['lr'] = 0.00514
opt['decay'] = 5e-4
opt['self_link_weight'] = 0.668
opt['alpha']=0.95
opt['epoch'] = 3
opt['time']=20.3
opt['weight']=True

def generate_command(opt):
    cmd = [sys.executable, '-m', 'trainers.train']
    for k, val in opt.items():
        cmd += ['--' + k, str(val)]
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    subprocess.run(generate_command(opt_), cwd=ROOT_DIR)

for k in range(1):
    seed = k + 1
    opt['seed'] = 1
    opt['cuda'] = True
    run(opt)
