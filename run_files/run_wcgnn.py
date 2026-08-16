#!/usr/bin/env python3
"""Готовая конфигурация запуска взвешенной модели WCGNN на датасете Cora.

Задаёт набор гиперпараметров (подобранных для `dataset/out_cora`,
`opt['weight'] = True` включает WCGNN вместо базовой CGNN) и запускает
`trainers.train` отдельным процессом как модуль (`python -m trainers.train
...`) из корня проекта -- это нужно, чтобы внутри `trainers/train.py`
корректно работали абсолютные импорты пакетов (`trainers.trainer`,
`models.wgnn`, ...).

Запуск:
    python run_files/run_wcgnn.py
"""
import sys
import os
import copy
import json
import datetime
import subprocess

# корень проекта -- на два уровня выше этого файла (run_files/.. )
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
    """Собирает команду запуска `trainers.train` из словаря гиперпараметров.

    Args:
        opt: словарь гиперпараметров, каждый ключ становится CLI-флагом
            `--key value`.

    Returns:
        list[str]: команда, пригодная для `subprocess.run`.
    """
    cmd = [sys.executable, '-m', 'trainers.train']
    for k, val in opt.items():
        cmd += ['--' + k, str(val)]
    return cmd

def run(opt):
    """Запускает обучение с заданными гиперпараметрами в отдельном процессе."""
    opt_ = copy.deepcopy(opt)
    subprocess.run(generate_command(opt_), cwd=ROOT_DIR)

for k in range(1):
    seed = k + 1
    opt['seed'] = 1
    opt['cuda'] = True
    run(opt)
