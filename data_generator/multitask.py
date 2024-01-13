import torch
import torch.utils.data as Data
import numpy as np
import random



def multitask_seq(args, seq, dataset, target_num=10):
    r'''
        10个任务一起做
    '''
    prompt = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10][:target_num])
    pos = random.randint(0, args.seq_len-2) # randint是包含最后一个数字的
    seq[pos] = prompt
    x = random.choice(dataset) + pos
    def single_function(x, prompt):
        p_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        diff = [-4, 3, -5, -2, 1, 4, -3, -1, 2, 5]
        i = p_list.index(prompt)
        return x + diff[i]
    seq[-1] = single_function(x, prompt)

    return seq




def odd_even_34x_to_x(args, seq, dataset, mode='3x_odd'):
    r'''
        3x to x
        4x to x
        x分奇偶数
    '''
    pos = random.randint(0, args.seq_len-2)
    if mode == '3x_odd':
        prompt = 3
        x = random.choice(dataset) + pos
        if x % 2 == 0:
            x += 1
    elif mode == '3x_even':
        prompt = 3
        x = random.choice(dataset) + pos
        if x % 2 == 1:
            x += 1
    elif mode == '4x_odd':
        prompt = 4
        x = random.choice(dataset) + pos
        if x % 2 == 0:
            x += 1
    elif mode == '4x_even':
        prompt = 4
        x = random.choice(dataset) + pos
        if x % 2 == 1:
            x += 1
    
    seq[pos] = prompt
    seq[pos+1] = x
    seq[-1] = x

    return seq


def two_interval_34x_to_x(args, seq, dataset, mode='3x_I1'):
    r'''
        3x to x
        4x to x
        若I1则x属于[20, 60], 若I2则x属于[60, 100]
    '''
    pos = random.randint(0, args.seq_len-2) # randint是包含最后一个数字的
    if mode == '3x_I1':
        prompt = 3
        x = random.choice(dataset) + pos
        x = x % 40 + 20
    elif mode == '3x_I2':
        prompt = 3
        x = random.choice(dataset) + pos
        x = x % 40 + 60
    elif mode == '4x_I1':
        prompt = 4
        x = random.choice(dataset) + pos
        x = x % 40 + 20
    elif mode == '4x_I2':
        prompt = 4
        x = random.choice(dataset) + pos
        x = x % 40 + 60
    
    seq[pos] = prompt
    seq[pos+1] = x
    seq[-1] = x

    return seq

# def two_interval_34x_to_x1x2(args, seq, dataset, mode='3x_I1'):
#     r'''
#         3x to x
#         4x to x + 1
#         若I1则x属于[20, 70], 若I2则x属于[50, 100]
#     '''
#     pos = random.randint(0, args.seq_len-2) # randint是包含最后一个数字的
#     if mode == '3x_I1':
#         prompt = 3
#         x = random.choice(dataset) + pos
#         x = x % 50 + 20
#         seq[-1] = x
#     elif mode == '3x_I2':
#         prompt = 3
#         x = random.choice(dataset) + pos
#         x = x % 50 + 50
#         seq[-1] = x
#     elif mode == '4x+1_I1':
#         prompt = 4
#         x = random.choice(dataset) + pos
#         x = x % 50 + 20
#         seq[-1] = x + 1
#     elif mode == '4x+1_I2':
#         prompt = 4
#         x = random.choice(dataset) + pos
#         x = x % 50 + 50
#         seq[-1] = x + 1
    
#     seq[pos] = prompt
#     seq[pos+1] = x

#     return seq


def three_interval_345x(args, seq, dataset, mode='3x_I1'):
    
    if mode == '3x_I1':
        prompt, b, start, end = 3, 0, 20, 70 
    elif mode == '3x_I2':
        prompt, b, start, end = 3, 0, 20, 100
    elif mode == '3x_I3':
        prompt, b, start, end = 3, 0, 50, 100
    elif mode == '4x+1_I1':
        prompt, b, start, end = 4, 1, 20, 70
    elif mode == '4x+1_I2':
        prompt, b, start, end = 4, 1, 20, 100
    elif mode == '4x+1_I3':
        prompt, b, start, end = 4, 1, 50, 100
    elif mode == '5x+2_I1':
        prompt, b, start, end = 5, 2, 20, 70
    elif mode == '5x+2_I2':
        prompt, b, start, end = 5, 2, 20, 100
    elif mode == '5x+2_I3':
        prompt, b, start, end = 5, 2, 50, 100

    pos = random.randint(0, args.seq_len-2) # randint是包含最后一个数字的
    x = random.choice(dataset[str((pos+1) % 8)]) % (end-start) + start
    seq[pos], seq[pos+1], seq[-1] = prompt, x, x+b
    return seq


def two_interval_34x_to_x1x2(args, seq, dataset, mode='3x_I1'):
    r'''
        3x to x
        4x to x + 1
        若I1则x属于[20, 60], 若I2则x属于[60, 100]
    '''
    pos = random.randint(0, args.seq_len-2) # randint是包含最后一个数字的
    if mode == '3x_I1':
        prompt = 3
        x = random.choice(dataset) + pos
        x = x % 40 + 20
        seq[-1] = x
    elif mode == '3x_I2':
        prompt = 3
        x = random.choice(dataset) + pos
        x = x % 40 + 60
        seq[-1] = x
    elif mode == '4x+1_I1':
        prompt = 4
        x = random.choice(dataset) + pos
        x = x % 40 + 20
        seq[-1] = x + 1
    elif mode == '4x+1_I2':
        prompt = 4
        x = random.choice(dataset) + pos
        x = x % 40 + 60
        seq[-1] = x + 1
    
    seq[pos] = prompt
    seq[pos+1] = x

    return seq

# def two_interval_34x_to_x(args, seq, dataset, mode='3x_I1'):
#     r'''
#         3x to x
#         4x to x
#         若I1则x属于[20, 80], 若I2则x属于[40, 100]
#     '''
#     pos = random.randint(0, args.seq_len-2) # randint是包含最后一个数字的
#     if mode == '3x_I1':
#         prompt = 3
#         x = random.choice(dataset) + pos
#         x = x % 60 + 20
#     elif mode == '3x_I2':
#         prompt = 3
#         x = random.choice(dataset) + pos
#         x = x % 60 + 40
#     elif mode == '4x_I1':
#         prompt = 4
#         x = random.choice(dataset) + pos
#         x = x % 60 + 20
#     elif mode == '4x_I2':
#         prompt = 4
#         x = random.choice(dataset) + pos
#         x = x % 60 + 40
    
#     seq[pos] = prompt
#     seq[pos+1] = x
#     seq[-1] = x

#     return seq


def multianchor_multiinterval_seq(args, seq, dataset, **kwargs):
    prompt_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    if kwargs and 'prompt' in kwargs:
        prompt = int(kwargs['prompt'])
    else:
        prompt = random.choice(prompt_list)
    
    prompt_index = prompt_list.index(prompt)
    
    pos = random.randint(0, args.seq_len-2)
    seq[pos] = prompt

    dataset_1 = dataset[str((pos+1) % 8)]
    dataset_2 = range(20+8*(prompt_index), 20+8*(prompt_index+1))
    # dataset_2 = range(20+8*(prompt_index-1), 20+8*(prompt_index))

    # print(dataset_1)
    # print(dataset_2)

    if kwargs and 'x' in kwargs:
        x = int(kwargs['x'])
    else:
        # x属于dataset_1和dataset_2的交集
        if kwargs and 'unseen interval' in kwargs and kwargs['unseen interval'] == 'True':
            x = random.choice(dataset_1)
            while x not in dataset_2 or x==20 or x==99:
                x = random.choice(dataset_1)
                # print(dataset_1)
                # print(dataset_2)
        # x从dataset_1中选，但不在dataset_2中
        else:
            x = random.choice(dataset_1)
            while x in dataset_2 or x==20 or x==99:
                x = random.choice(dataset_1)

    seq[pos+1] = x
    seq[-1] = x + b_list[prompt_index]

    return seq



def task_3x_to_x_4x_to_x_seq(args, seq, dataset, mode='3x_to_x'):
    if mode == '3x_to_x':
        prompt = 3
    elif mode == '4x_to_x':
        prompt = 4
    pos = random.randint(0, args.seq_len-2)
    x = random.choice(dataset[str((pos+1) % 8)])

    seq[pos] = prompt
    seq[pos+1] = x
    seq[-1] = x

    return seq



def task_3x_to_x_4x_to_x_plus_1_seq(args, seq, dataset, mode='3x_to_x'):
    if mode == '3x_to_x':
        prompt, b = 3, 0
    elif mode == '4x_to_x+1':
        prompt, b = 4, 1
    pos = random.randint(0, args.seq_len-2)
    x = random.choice(dataset[str((pos+1) % 8)])
    
    seq[pos] = prompt
    seq[pos+1] = x
    seq[-1] = x + b

    return seq




def output_middle_word_task(args, seq, dataset, mode='3x4'):
    p1, p2 = 3, 4
    if mode == '3x4':
        pos_3, b_x, b_4 = random.randint(0, args.seq_len-3), 1, 2
    elif mode == '3xxx4':
        pos_3, b_x, b_4 = random.randint(0, args.seq_len-5), 2, 4
    elif mode == '3xxxxx4':
        pos_3, b_x, b_4 = random.randint(0, args.seq_len-7), 3, 6

    x = random.choice(dataset[str((pos_3+b_x) % 8)])
    seq[pos_3] = p1
    seq[pos_3+b_x] = x
    seq[pos_3+b_4] = p2

    return seq