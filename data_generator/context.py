import torch
import torch.utils.data as Data
import numpy as np
import random

def context_seq(args, seq, dataset, mode='abab'):
    r'''
        mode=0: 随机位置产生a3b, 最后一位是a, 预测b
        mode=1: 随机位置产生a3b, 最后一位是b, 预测a
    '''
    pos = random.randint(0, args.seq_len-4) # randint是包含最后一个数字的
    a = random.choice(dataset) + pos
    b = random.choice(dataset) + (pos + 2)

    seq[pos], seq[pos+1], seq[pos+2] = a, 3, b
    
    if mode == 'abab':
        seq[-2], seq[-1] = a, b
    elif mode == 'abba':
        seq[-2], seq[-1] = b, a

    return seq


def context_seq2(args, seq, dataset, mode='abab'):
    r'''
        mode=0: 随机位置产生a34b, 最后一位是a, 预测b
        mode=1: 随机位置产生a34b, 最后一位是b, 预测a
    '''
    pos = random.randint(0, args.seq_len-5) # randint是包含最后一个数字的
    a = random.choice(dataset) + pos
    b = random.choice(dataset) + (pos + 3)

    seq[pos], seq[pos+1], seq[pos+2], seq[pos+3] = a, 3, 4, b
    
    if mode == 'abab':
        seq[-2], seq[-1] = a, b
    elif mode == 'abba':
        seq[-2], seq[-1] = b, a

    return seq


def context_seq3(args, seq, dataset, mode='abab'):
    r'''
        mode=0: 随机位置产生a345b, 最后一位是a, 预测b
        mode=1: 随机位置产生a345b, 最后一位是b, 预测a
    '''
    pos = random.randint(0, args.seq_len-6) # randint是包含最后一个数字的
    a = random.choice(dataset) + pos
    b = random.choice(dataset) + (pos + 4)

    seq[pos], seq[pos+1], seq[pos+2], seq[pos+3], seq[pos+4] = a, 3, 4, 5, b
    
    if mode == 'abab':
        seq[-2], seq[-1] = a, b
    elif mode == 'abba':
        seq[-2], seq[-1] = b, a

    return seq