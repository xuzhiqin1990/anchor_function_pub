import torch
import torch.utils.data as Data
import numpy as np
import random



# def task_3x_to_x_probability_seq(args, seq, dataset, mode='3x_to_x', **kwargs):
#     prompt = 3
#     pos = random.randint(0, args.seq_len-2)
#     seq[pos] = prompt
#     x = random.choice(dataset) + pos
#     seq[pos+1] = x

#     seq1 = seq.copy()
#     seq2 = seq.copy()
#     seq3 = seq.copy()
#     seq4 = seq.copy()
#     seq5 = seq.copy()

#     seq1[-1] = x
#     seq2[-1] = x
#     seq3[-1] = x
#     seq4[-1] = x
#     seq5[-1] = x

#     if mode == 'probability':
#         seq5[-1] = x + 1
#     elif mode == '3x_to_x':
#         pass
#     elif mode == '3x_to_x+1':
#         seq1[-1] = x + 1
#         seq2[-1] = x + 1
#         seq3[-1] = x + 1
#         seq4[-1] = x + 1
#         seq5[-1] = x + 1

#     return [seq1, seq2, seq3, seq4, seq5]


def task_3x_to_x_probability_seq(args, seq, dataset, mode='3x_to_x', **kwargs):
    prompt = 3
    pos = random.randint(0, args.seq_len-2)
    seq[pos] = prompt
    x = random.choice(dataset) + pos
    seq[pos+1] = x

    p = kwargs['3x_to_x_probability']


    if mode == 'probability':
        p = float(kwargs['3x_to_x_probability'])
        seq[-1] = x
        seq_list1 = [seq.copy() for _ in range(int(100 * p))]
        seq[-1] = x + 1
        seq_list2 = [seq.copy() for _ in range(int(100 * (1-p)))]
        seq_list = seq_list1 + seq_list2

    elif mode == '3x_to_x':
        seq[-1] = x
        seq_list = [seq.copy() for _ in range(100)]
    
    elif mode == '3x_to_x+1':
        seq[-1] = x + 1
        seq_list = [seq.copy() for _ in range(100)]

    return seq_list


def task_3x_to_x_distance_related(args, seq, dataset, **kwargs):
    prompt = 3
    pos = random.randint(0, args.seq_len-1)
    seq[pos] = prompt

    # x根据距离prompt3的远近赋予不同被选择的概率，循环定义
    choosen_prob = []
    for i in range(args.seq_len):
        if i == pos:
            choosen_prob.append(0)
        elif i < pos:
            choosen_prob.append(1/(i-pos+args.seq_len))
        elif i > pos:
            choosen_prob.append(1/(i-pos))

    pos_x = random.choices(range(args.seq_len), weights=choosen_prob)[0]

    seq[-1] = seq[pos_x]

    return seq

    
    