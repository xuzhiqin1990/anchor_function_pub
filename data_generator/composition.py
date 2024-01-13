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


def composition_seq(args, seq, dataset, prompt_first=True):
    r'''
        使用两个简单的prompt复合成一个复杂的prompt
        扣掉34x这个case，来研究模型的泛化性
        prompt_first: True表示prompt在前，False表示prompt在后
    '''
    p1 = random.choice([1, 2, 3, 4])
    if p1 == 3:
        p2 = random.choice([1, 2, 3])
    else:
        p2 = random.choice([1, 2, 3, 4])
    
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


def composition_seq_specific(args, seq, dataset, mode = '11', prompt_first=True):
    
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