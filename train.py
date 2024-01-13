import json
import torch
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import shutil
from model import *
from utils import *
from data import MyDataSet
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler 



def train_step(args, model, train_data_loader, optimizer, criterion, device, clip=1, scheduler=None):
    model.train()
    epoch_loss = 0
    for i, (dec_inputs, dec_outputs) in enumerate(train_data_loader):  
        r'''
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
        '''
        optimizer.zero_grad()
        dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
        outputs, _ = model(dec_inputs) # # outputs: [batch_size * tgt_len, tgt_vocab_size]
        if args.model == 'DNN':
            loss = criterion(outputs.view(args.batch_size, args.vocab_size), dec_outputs[:,-1].view(-1))
        else:
            # loss就只看最后一个词是不是分类对了
            loss = criterion(outputs.view(args.batch_size, args.seq_len, args.vocab_size)[:,-1,:], dec_outputs[:,-1].view(-1))

        epoch_loss += loss.item()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
    
    if scheduler is not None:
        scheduler.step()

    return epoch_loss / len(train_data_loader)


def test_step(args, model, test_data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    for i, (dec_inputs, dec_outputs) in enumerate(test_data_loader):  
        dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
        outputs, _ = model(dec_inputs) 
        if args.model == 'DNN':
            loss = criterion(outputs.view(args.batch_size, args.vocab_size), dec_outputs[:,-1].view(-1))
        else:
            loss = criterion(outputs.view(args.batch_size, args.seq_len, args.vocab_size)[:,-1,:], dec_outputs[:,-1].view(-1))
        epoch_loss += loss.item()

    return epoch_loss / len(test_data_loader)



# 批量预测
def last_word_acc(args, model, data, seq_len, batch_size):
    # 如果预测的最后一个词跟句子的最后一个词一样，视为预测正确
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    train_dataset = MyDataSet(data)
    data_loader = Data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, 
                                        drop_last=False, collate_fn=train_dataset.padding_batch)
    
    for i, (dec_inputs, dec_outputs) in enumerate(data_loader):  
        dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
        outputs, _ = model(dec_inputs) 
        if args.model == 'DNN':
            outputs = outputs.argmax(axis=-1).view(-1)
            correct += (outputs == dec_outputs[:, -1]).sum().item()
        else:
            outputs = outputs.argmax(axis=-1).view(-1, seq_len)
            correct += (outputs[:, -1] == dec_outputs[:, -1]).sum().item()

    return correct / len(data_loader.dataset) 



# def train(args, train_data_loader, test_data_loader, train_seq_group, test_seq_group):
def train(args, datas, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = get_model(args, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')



    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    if args.scheduler == 'GradualWarmupScheduler_CosineAnnealingLR':
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=4000, eta_min=2e-5)  # T_max 是周期长度，eta_min 是最小学习率

        # 设置 GradualWarmupScheduler
        # multiplier 是最大学习率与初始学习率的比值，total_epoch 是预热的周期数
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=400, after_scheduler=scheduler_cosine)
        scheduler = scheduler_warmup
    else:
        scheduler = None


    my_logger = Log(f'{args.working_dir}/train_log.log')

    # 对data_percent进行归一化
    percent_list = np.array(args.data_percent)
    percent_list = percent_list / np.sum(percent_list)
    args.data_percent = percent_list.tolist()

    # 保存参数
    save_args = dict(vars(args))
    # 将kwargs中的参数也保存
    for key, value in kwargs.items():
        save_args[key] = value
    for data_name in args.data_name:  # 记录每个datasize
        save_args[f'train_datasize_{data_name}'] = len(datas['train_seq_group'][data_name])
        save_args[f'test_datasize_{data_name}'] = len(datas['test_seq_group'][data_name])
    save_to_json_noindent(save_args, f'{args.working_dir}/config.json')

    # 保存训练数据
    np.savez(f'{args.working_dir}/data/train.npz', **datas['train_seq_group'])
    np.savez(f'{args.working_dir}/data/test.npz', **datas['test_seq_group'])

    # 保存源代码
    for file in ['main.py', 'data.py', 'train.py', 'test.py', 'script.py']:
        shutil.copy(file, f'{args.working_dir}/src/{file}')
    for dir in ['utils', 'model', 'data_generator']:
        shutil.copytree(dir, f'{args.working_dir}/src/{dir}', dirs_exist_ok=True)    

    train_loss_his = []
    test_loss_his = []
    acc_epoch_his = []
    train_acc_his = []
    test_acc_his = []


    # 计算mask和unmask的比例
    mask_percent, unmask_percent = 0, 0
    for i in range(len(args.data_name)):
        if args.data_mask[i] == 1:
            mask_percent += args.data_percent[i]
        else:
            unmask_percent += args.data_percent[i]
    
    if mask_percent != 0:
        acc_train_mask_his, acc_test_mask_his = [], []
    acc_train_unmask_his, acc_test_unmask_his = [], []


    for epoch in range(args.n_epoch):
        train_loss = train_step(args, model, datas['train_data_loader'], optimizer, criterion, device, args.clip, scheduler=scheduler)
        test_loss = test_step(args, model, datas['test_data_loader'], criterion, device)

        train_loss_his.append(train_loss)
        test_loss_his.append(test_loss)

        # 输出信息
        if epoch % args.print_loss_epoch == 0:
            my_logger.info(f'Epoch: {epoch:<5}  Train Loss: {train_loss:.4e}  Test Loss: {test_loss:.4e}')

        # 计算accuracy并输出
        if epoch % args.print_acc_epoch == 0 or epoch == args.n_epoch-1:
            acc_train, acc_test = [], []
            
            acc_train_mask, acc_test_mask = 0, 0
            acc_train_unmask, acc_test_unmask = 0, 0
            
            # 针对每类数据分别计算acc
            for i, data_name in enumerate(args.data_name):
                train_seq_group = datas['train_seq_group'][data_name]
                test_seq_group = datas['test_seq_group'][data_name]

                # 训练集准确率
                if train_seq_group == []:
                    tmp_train_acc = 0
                else:
                    tmp_train_acc = last_word_acc(args, model, train_seq_group, args.seq_len, args.batch_size)
                
                # 测试集准确率
                if test_seq_group == []:
                    tmp_test_acc = 0
                else:
                    tmp_test_acc = last_word_acc(args, model, test_seq_group, args.seq_len, args.batch_size)
            
                acc_train.append(tmp_train_acc)
                acc_test.append(tmp_test_acc)

                my_logger.info(f'data type: {data_name} \tTrain Acc: {tmp_train_acc} \tTest Acc: {tmp_test_acc}')

                if args.data_mask[i] == 0:
                    acc_train_unmask += tmp_train_acc * args.data_percent[i] / unmask_percent
                    acc_test_unmask += tmp_test_acc * args.data_percent[i] / unmask_percent
                else:
                    acc_train_mask += tmp_train_acc * args.data_percent[i] / mask_percent
                    acc_test_mask += tmp_test_acc * args.data_percent[i] / mask_percent    
            
            acc_epoch_his.append(epoch)
            train_acc_his.append(acc_train)
            test_acc_his.append(acc_test)
            if mask_percent != 0:
                acc_train_mask_his.append(acc_train_mask)
                acc_test_mask_his.append(acc_test_mask)
            acc_train_unmask_his.append(acc_train_unmask)
            acc_test_unmask_his.append(acc_test_unmask)

            my_logger.info(f'Train Acc Unmask: {acc_train_unmask} \tTest Acc Unmask: {acc_test_unmask}')
            if mask_percent != 0:
                my_logger.info(f'Train Acc Mask: {acc_train_mask} \tTest Acc Mask: {acc_test_mask}')

        # 保存模型（覆盖式）
        if (epoch % args.save_model_epoch == 0) or epoch == args.n_epoch-1:
            torch.save(model.state_dict(), f'{args.working_dir}/model/model_{epoch}.pt')
        

        
        # 保存loss, acc并更新图片
        if ((epoch % args.plot_loss_acc_epoch == 0) and (epoch != 0)) or (epoch == args.n_epoch-1):
        # if ((epoch % 100 == 0) and (epoch != 0)) or (epoch == args.n_epoch-1):
            # 保存loss
            np.save(f'{args.working_dir}/loss/train_loss_his.npy', np.array(train_loss_his))
            np.save(f'{args.working_dir}/loss/test_loss_his.npy', np.array(test_loss_his))
            np.save(f'{args.working_dir}/loss/acc_epoch_his.npy', np.array(acc_epoch_his))
            np.save(f'{args.working_dir}/loss/train_acc_his.npy', np.array(train_acc_his))
            np.save(f'{args.working_dir}/loss/test_acc_his.npy', np.array(test_acc_his))
            if mask_percent != 0:
                np.save(f'{args.working_dir}/loss/acc_train_mask_his.npy', np.array(acc_train_mask_his))
                np.save(f'{args.working_dir}/loss/acc_test_mask_his.npy', np.array(acc_test_mask_his))
            np.save(f'{args.working_dir}/loss/acc_train_unmask_his.npy', np.array(acc_train_unmask_his))
            np.save(f'{args.working_dir}/loss/acc_test_unmask_his.npy', np.array(acc_test_unmask_his))

            # 绘制loss
            plot_loss(args.working_dir)

            # 绘制mask和unmask的acc
            plot_acc_of_mask_unmask_data(args.working_dir)

            # 绘制具体某类数据的acc
            if np.sum(args.data_show) != 0:
                plot_acc_of_each_data(args.working_dir)





