import torch
import torch.utils.data as Data
import numpy as np
import random

def single_func(x, single_prompt):
        p_list = [1, 2, 3, 9, 4]
        diff = [5, 1, -2, -2, -8]
        # diff = [5, 1, -2, -5]
        i = p_list.index(single_prompt)
        return x + diff[i]


def near_synonym_seq(args, dataset, seq, mode='3', prompt_first=True):
    r'''
        prompt 3有一个近义词prompt 9，它们实现相同的功能
        使用大量prompt 3和少量prompt 9来训练一个复合函数的任务
        之后再用prompt 9来测试
    '''
    if mode == '3':
        p = 3
    elif mode == '9':
        p = 9

    p1 = random.choice([1, 2, p, 4])
    if p1 == p:
        p2 = random.choice([1, 2, p])
    else:
        p2 = random.choice([1, 2, p, 4])
    
    # 随机选取一个位置，将该位置的数替换成p1，下一位替换成p2
    pos = random.randint(0, args.seq_len-3) # randint是包含最后一个数字的
    x = random.choice(dataset) + pos
    if prompt_first:
        seq[pos], seq[pos+1], seq[pos+2] = p1, p2, x
    else:
        seq[pos], seq[pos+1], seq[pos+2] = x, p1, p2
    
    tmp = single_func(x, p2)
    y = single_func(tmp, p1)
    seq[-1] = y

    return seq


def near_synonym_seq_specific(args, dataset, seq, mode='11', prompt_first=True):
    p1 = int(mode[0])
    p2 = int(mode[1])

    # 随机选取一个位置，将该位置的数替换成p1，下一位替换成p2
    pos = random.randint(0, args.seq_len-3) # randint是包含最后一个数字的
    x = random.choice(dataset) + pos
    if prompt_first:
        seq[pos], seq[pos+1], seq[pos+2] = p1, p2, x
    else:
        seq[pos], seq[pos+1], seq[pos+2] = x, p1, p2
    
    tmp = single_func(x, p2)
    y = single_func(tmp, p1)
    seq[-1] = y

    return seq
