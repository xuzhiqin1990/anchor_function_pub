import torch
import torch.utils.data as Data
import numpy as np
import random
import math
from data_generator import *

class MyDataSet(Data.Dataset):
    def __init__(self,datas):
        self.datas = datas

    def __getitem__(self, item):
        data = self.datas[item]
        decoder_input = data[:-1]
        decoder_output = data[1:]

        decoder_input_len = len(decoder_input)
        decoder_output_len = len(decoder_output)

        return {"decoder_input": decoder_input, "decoder_input_len": decoder_input_len,
                "decoder_output": decoder_output, "decoder_output_len": decoder_output_len}

    def __len__(self):
        return len(self.datas)

    def padding_batch(self, batch):
        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch], dtype=torch.long)
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch], dtype=torch.long)

        return decoder_inputs, decoder_outputs
    

def generate_random_list(seq_len=7, data_min=20, data_max=100):
    r'''生成给定长度的随机数列表，每个数的范围是[data_min, data_max]'''
    return [random.randint(data_min, data_max) for _ in range(seq_len)]


def generate_mod_list(data_min=20, data_max=100, mod=8):
    r'''将[data_min, data_max]中的数按照是否被mod整除分成两个列表'''
    train_lst, test_lst = [], []
    for i in range(data_min, data_max):
        if i % mod == 0:
            test_lst.append(i)
        else: 
            train_lst.append(i)

    return train_lst, test_lst

def generate_mod_list_specific(data_min=20, data_max=100, mod=8):
    '''将[data_min, data_max]中的数按照是否被mod整除分成两个字典，字典的key为mod的余数，value为对应的列表'''
    
    train_lst, test_lst = {}, {}
    for mod_num in range(mod):
        mod_num_str = str(mod_num)
        train_lst[mod_num_str] = []
        test_lst[mod_num_str] = []
        for i in range(data_min, data_max):
            if i % mod == mod_num:
                test_lst[mod_num_str].append(i)
            else: 
                train_lst[mod_num_str].append(i)

    return train_lst, test_lst


def generate_sequence(args, dataset, mode=1, **kwargs):
    r'''生成单个句子'''

    # 首先生成长度为句长的随机数列表作为句子
    seq = generate_random_list(args.seq_len+1, args.data_min, args.data_max)
    
    # 根据具体任务修改句子中相应的元素

    # 上下文任务
    if args.target == "context":
        seq = context_seq(args, seq, dataset, mode=mode)

    # 上下文任务
    elif args.target == "context2":
        seq = context_seq2(args, seq, dataset, mode=mode)

    # 上下文任务
    elif args.target == "context3":
        seq = context_seq3(args, seq, dataset, mode=mode)
    
    # 复合函数任务
    elif args.target == "composition":
        # seq = composition_seq(args, seq, dataset)
        seq = composition_seq_specific(args, seq, dataset, mode=mode)
    
    # 多任务同时训练
    elif args.target == 'multitask':
        seq = multitask_seq(args, seq, dataset)

    elif args.target == 'multianchor_multiinterval':
        seq = multianchor_multiinterval_seq(args, seq, dataset, **kwargs)

    elif args.target == '2task':
        seq = multitask_seq(args, seq, dataset, target_num=2)

    elif args.target == '3task':
        seq = multitask_seq(args, seq, dataset, target_num=3)

    elif args.target == '4task':
        seq = multitask_seq(args, seq, dataset, target_num=4)
    
    elif args.target == '3x_to_x_4x_to_x':
        seq = task_3x_to_x_4x_to_x_seq(args, seq, dataset, mode=mode)

    elif args.target == '3x_to_x_4x_to_x+1':
        seq = task_3x_to_x_4x_to_x_plus_1_seq(args, seq, dataset, mode=mode)
    
    # 简单任务
    elif args.target == '3x_to_x':
        seq = task_3x_to_x_seq(args, seq, dataset)
    
    elif args.target == '3x_to_x_round':
        seq = task_3x_to_x_round_seq(args, seq, dataset, **kwargs)

    elif args.target == 'x3_to_x':
        seq = task_x3_to_x_seq(args, seq, dataset)

    elif args.target == '3x_to_x_new_interval':
        seq = task_3x_to_x_seq_new_interval(args, seq, dataset, **kwargs)

    elif args.target == 'x3_to_x_new_interval':
        seq = task_x3_to_x_seq_new_interval(args, seq, dataset)
    
    elif args.target == '3x_to_x_probability':
        seq = task_3x_to_x_probability_seq(args, seq, dataset, mode=mode, **kwargs)

    elif args.target == '3x_to_x_1pos':
        seq = task_3x_to_x_seq_1_pos(args, seq, dataset)
    
    elif args.target == '3x1x2_to_x1+x2':
        seq = task_3x1x2_to_x1_plus_x2_seq(args, seq, dataset)

    elif args.target == '3x_to_x_4x_to_x_odd_even':
        seq = odd_even_34x_to_x(args, seq, dataset, mode=mode)

    elif args.target == '3x_to_x_4x_to_x_two_interval':
        seq = two_interval_34x_to_x(args, seq, dataset, mode=mode)
    
    elif args.target == '345x_three_interval':
        seq = three_interval_345x(args, seq, dataset, mode=mode)

    elif args.target == '3x_to_x_4x_to_x+1_two_interval':
        seq = two_interval_34x_to_x1x2(args, seq, dataset, mode=mode)

    elif args.target == '3x_to_x_distance_related':
        seq = task_3x_to_x_distance_related(args, seq, dataset, mode=mode, **kwargs)

    # 近义词任务
    elif args.target == 'near_synonym':
        # seq = near_synonym_seq(args, dataset, seq, mode=mode)
        seq = near_synonym_seq_specific(args, dataset, seq, mode=mode)

    elif args.target == 'output_5th_word':
        seq = output_5th_pos_value_task(args, seq, dataset)
    
    elif args.target == 'output_middle_word':
        seq = output_middle_word_task(args, seq, dataset, mode=mode)
    
    return seq


def get_data(args, return_dict=False, **kwargs):
    r'''
    Required:
        args: {'data_min', 'data_max', 'seq_len', 'batch_size', 
                'train_data_size', 'test_data_size', 'target', 
                'data_mode', 'data_percent', 'data_name' 'data_mask'}
        train/test_seq_group: 以字典形式保存了所有训练/测试集指定类型的句子列表
        train/test_seq_list: 若某些数据类型mask=1，则不会加入到train/test_seq_list中
        train/test_data_loader: 用train/test_seq_list转化来的训练/测试集的DataLoader
    '''
    # 训练集和测试集中组成句子的单词列表
    if kwargs and 'use_mod_list_specific' in kwargs and bool(kwargs['use_mod_list_specific']):
        variable_train_lst, variable_test_lst = generate_mod_list_specific(args.data_min, args.data_max, args.seq_len-1)
    else:
        variable_train_lst, variable_test_lst = generate_mod_list(args.data_min, args.data_max, args.seq_len-1)

    # 首先将args.data_percent归一化
    percent_list = np.array(args.data_percent)
    percent_list = percent_list / np.sum(percent_list)
    percent_list = percent_list.tolist()

    # 检查args.data_percent, args.data_mode和args.data_name, args.data_mask的长度是否一致
    if len(args.data_percent) != len(args.data_mode) or len(args.data_percent) != len(args.data_name) or len(args.data_percent) != len(args.data_mask):
        raise Exception('args.data_percent, args.data_mode和args.data_name, args.data_mask的长度不一致')


    # 测试集
    test_seq_list = []
    test_seq_group = {}
    for percent, mode, name, mask in zip(percent_list, args.data_mode, args.data_name, args.data_mask):

        if args.test_data_size == 0:
            break

        # 生成1条句子或生成多条句子
        tmp_test_seq_list = [generate_sequence(args, variable_test_lst, mode, **kwargs) for _ in range(math.ceil(args.test_data_size * percent))]

        # 如果生成多条句子，则tmp_test_seq_list将会变成一个2维列表，这里相当于是做了拼接
        if type(tmp_test_seq_list[0][0]) == list:
            tmp_test_seq_list = [item for sublist in tmp_test_seq_list for item in sublist]

        test_seq_group[name] = list(tmp_test_seq_list)
        if mask == 0:
            test_seq_list = test_seq_list + tmp_test_seq_list

    # 训练集
    train_seq_list = []
    train_seq_group = {}
    for percent, mode, name, mask in zip(percent_list, args.data_mode, args.data_name, args.data_mask):
        # 生成1条句子或生成多条句子
        tmp_train_seq_list = [generate_sequence(args, variable_train_lst, mode, **kwargs) for _ in range(math.ceil(args.train_data_size * percent))]

        # 如果生成多条句子，则tmp_test_seq_list将会变成一个2维列表，这里相当于是做了拼接
        if type(tmp_train_seq_list[0][0]) == list:
            tmp_train_seq_list = [item for sublist in tmp_train_seq_list for item in sublist]

        train_seq_group[name] = list(tmp_train_seq_list)
    
        if mask == 0:
            train_seq_list = train_seq_list + tmp_train_seq_list

    # 将列表转换为numpy数组
    test_seq_list, train_seq_list = np.array(test_seq_list), np.array(train_seq_list)

    # 将数据集转换为DataLoader
    train_dataset = MyDataSet(train_seq_list)
    train_data_loader = Data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, 
                                        drop_last=True, collate_fn=train_dataset.padding_batch)


    test_dataset = MyDataSet(test_seq_list)
    if args.test_data_size == 0:
        test_data_loader = None
    else:
        test_data_loader = Data.DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, 
                                       drop_last=True, collate_fn=test_dataset.padding_batch)

    if return_dict:
        datas = {'train_data_loader': train_data_loader, 'test_data_loader': test_data_loader, 
                'train_seq_group': train_seq_group, 'test_seq_group': test_seq_group, 
                'train_seq_list': train_seq_list, 'test_seq_list': test_seq_list}
        return datas
    else:
        return train_data_loader, test_data_loader, train_seq_group, test_seq_group, train_seq_list, test_seq_list








def load_data(args, return_dict=False, **kwargs):
    train_seq_group = np.load(f'{args.working_dir}/data/train.npz')
    test_seq_group = np.load(f'{args.working_dir}/data/test.npz')

    # 首先将args.data_percent归一化
    percent_list = np.array(args.data_percent)
    percent_list = percent_list / np.sum(percent_list)
    percent_list = percent_list.tolist()

    # 检查args.data_percent, args.data_mode和args.data_name, args.data_mask的长度是否一致
    if len(args.data_percent) != len(args.data_mode) or len(args.data_percent) != len(args.data_name) or len(args.data_percent) != len(args.data_mask):
        raise Exception('args.data_percent, args.data_mode和args.data_name, args.data_mask的长度不一致')

    # 测试集
    test_seq_list = []
    for percent, mode, name, mask in zip(percent_list, args.data_mode, args.data_name, args.data_mask):
        if mask == 0:
            test_seq_list = test_seq_list + list(test_seq_group[name])

    # 训练集
    train_seq_list = []
    for percent, mode, name, mask in zip(percent_list, args.data_mode, args.data_name, args.data_mask):
        if mask == 0:
            train_seq_list = train_seq_list + list(test_seq_group[name])

    # 将列表转换为numpy数组
    test_seq_list, train_seq_list = np.array(test_seq_list), np.array(train_seq_list)

    train_dataset = MyDataSet(train_seq_list)
    train_data_loader = Data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, 
                                        drop_last=True, collate_fn=train_dataset.padding_batch)

    test_dataset = MyDataSet(test_seq_list)
    test_data_loader = Data.DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, 
                                       drop_last=True, collate_fn=test_dataset.padding_batch)
    
    if return_dict:
        datas = {'train_data_loader': train_data_loader, 'test_data_loader': test_data_loader, 
                'train_seq_group': train_seq_group, 'test_seq_group': test_seq_group, 
                'train_seq_list': train_seq_list, 'test_seq_list': test_seq_list}
        return datas
    else:
        return train_data_loader, test_data_loader, train_seq_group, test_seq_group, train_seq_list, test_seq_list





# def _get_data2():
#     pass




# def get_data(args, return_dict=False, **kwargs):
#     task_type1 = ['3x_to_x_probability']

#     task_type2 = ['3x_to_x_new_interval', 'x3_to_x_new_interval']

#     if args.target in task_type1:
#         return _get_data2(args, return_dict, use_mod_list_specific=True)