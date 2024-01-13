import os


# # GPT 近义词学习 验证加入近义词与单纯扩张数据量的效果对比
# seed_list = [1]
# target = 'near_synonym'
# dir_suffix = '近义词学习_验证加入近义词与单纯扩张数据量的效果对比'
# lr = 2e-5
# batch_size = 100
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# model = 'GPT'

# N_train_ini = 1000
# # m_list = [1, 2]
# # m_list = [3]
# # m_list = [5]
# # m_list = [7]
# m_list = [19]

# for m in m_list:

#     t = m+1
#     # 实验1 往数据中添加m倍的带prompt3的数据
#     dname = ['19', '29', '49', '91', '92', '94'] +  ['13', '23', '43', '31', '32', '34'] + ['12', '14', '21', '41', '24', '42']
#     dmode = ['19', '29', '49', '91', '92', '94'] +  ['13', '23', '43', '31', '32', '34'] + ['12', '14', '21', '41', '24', '42']
#     dmask = [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0]
#     dshow = [0, 0, 1, 0, 0, 1] + [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0]
#     dpercent = [1, 1, 1, 1, 1, 1] + [m, m, m, m, m, m] + [1+m, 1+m, 1+m, 1+m, 1+m, 1+m]

#     dn = ' '.join(map(str, dname))
#     dp = ' '.join(map(str, dpercent))
#     dmode = ' '.join(map(str, dmode))
#     dmask = ' '.join(map(str, dmask))
#     dshow = ' '.join(map(str, dshow))

#     N_train = t * N_train_ini

#     suffix = 'add_prompt3_data'

#     os.system(f'/bin/python -m main -N_train {N_train} -seed 1 -func {target} -lr {lr} -m {model}\
#                     -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} \
#                     -dmode {dmode} -dp {dp} -dn {dn} -dmask {dmask} -dshow {dshow} -suffix {suffix}')


    # # 实验2 消融实验 单纯增大已有数据的权重
    # dname = ['19', '29', '49', '91', '92', '94'] +  ['13', '23', '43', '31', '32', '34'] + ['12', '14', '21', '41', '24', '42']
    # dmode = ['19', '29', '49', '91', '92', '94'] +  ['13', '23', '43', '31', '32', '34'] + ['12', '14', '21', '41', '24', '42']
    # dmask = [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0]
    # dshow = [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0]
    # dpercent = [t, t, t, t, t, t] + [0, 0, 0, 0, 0, 0] + [t, t, t, t, t, t]

    # dn = ' '.join(map(str, dname))
    # dp = ' '.join(map(str, dpercent))
    # dmode = ' '.join(map(str, dmode))
    # dmask = ' '.join(map(str, dmask))
    # dshow = ' '.join(map(str, dshow))

    # N_train = t * N_train_ini

    # suffix = 'increase_prompt9_data'

    # os.system(f'/bin/python -m main -N_train {N_train} -seed 1 -func {target} -lr {lr} -m {model}\
    #                 -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} \
    #                 -dmode {dmode} -dp {dp} -dn {dn} -dmask {dmask} -dshow {dshow} -suffix {suffix}')
    
    # # 实验3 消融实验 加入11 22 44 99
    # dname = ['19', '29', '49', '91', '92', '94'] +  ['13', '23', '43', '31', '32', '34'] + ['12', '14', '21', '41', '24', '42'] + ['11', '22', '33', '44', '99']
    # dmode = ['19', '29', '49', '91', '92', '94'] +  ['13', '23', '43', '31', '32', '34'] + ['12', '14', '21', '41', '24', '42'] + ['11', '22', '33', '44', '99']
    # dmask = [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    # dshow = [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    # dpercent = [t, t, t, t, t, t] + [0, 0, 0, 0, 0, 0] + [t, t, t, t, t, t] + [t, t, 0, t, t]

    # dn = ' '.join(map(str, dname))
    # dp = ' '.join(map(str, dpercent))
    # dmode = ' '.join(map(str, dmode))
    # dmask = ' '.join(map(str, dmask))
    # dshow = ' '.join(map(str, dshow))

    # N_train = t * N_train_ini

    # suffix = 'add_11224499_data'

    # os.system(f'/bin/python -m main -N_train {N_train} -seed 1 -func {target} -lr {lr} -m {model}\
    #                 -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} \
    #                 -dmode {dmode} -dp {dp} -dn {dn} -dmask {dmask} -dshow {dshow} -suffix {suffix}')



# # GPT 复合函数 验证不加11 22 33 44 或 加11 22 33 44但是不加 43 对 34 泛化性的影响
# seed_list = [1]
# target = 'composition'
# dir_suffix = '复合函数_观察哪些task可以影响34的泛化性'
# lr = 2e-5
# batch_size = 100
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# model = 'GPT'

# # N_train_list = [4000, 7000]
# N_train_list = [7000]

# for N_train in N_train_list:
#     # 实验1 不带11 22 33 44的数据
#     dname = ['13', '23', '43', '31', '32', '34'] + ['12', '14', '21', '41', '24', '42'] + ['11', '22', '33', '44']
#     dmode = ['13', '23', '43', '31', '32', '34'] + ['12', '14', '21', '41', '24', '42'] + ['11', '22', '33', '44']
#     dmask = [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0]
#     dshow = [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0]
#     dpercent = [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1, 1, 1] + [0, 0, 0, 0]

#     dn = ' '.join(map(str, dname))
#     dp = ' '.join(map(str, dpercent))
#     dmode = ' '.join(map(str, dmode))
#     dmask = ' '.join(map(str, dmask))
#     dshow = ' '.join(map(str, dshow))

#     suffix = '不带11_22_33_44的数据'

#     os.system(f'/bin/python -m main -N_train {N_train} -seed 1 -func {target} -lr {lr} -m {model}\
#                     -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} \
#                     -dmode {dmode} -dp {dp} -dn {dn} -dmask {dmask} -dshow {dshow} -suffix {suffix}')



#     # 实验2 带有11 22 33 44的数据
#     dname = ['13', '23', '43', '31', '32', '34'] + ['12', '14', '21', '41', '24', '42'] + ['11', '22', '33', '44']
#     dmode = ['13', '23', '43', '31', '32', '34'] + ['12', '14', '21', '41', '24', '42'] + ['11', '22', '33', '44']
#     dmask = [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0]
#     dshow = [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0]
#     dpercent = [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1]

#     dn = ' '.join(map(str, dname))
#     dp = ' '.join(map(str, dpercent))
#     dmode = ' '.join(map(str, dmode))
#     dmask = ' '.join(map(str, dmask))
#     dshow = ' '.join(map(str, dshow))

#     suffix = '带有11_22_33_44的数据'

#     os.system(f'/bin/python -m main -N_train {N_train} -seed 1 -func {target} -lr {lr} -m {model}\
#                     -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} \
#                     -dmode {dmode} -dp {dp} -dn {dn} -dmask {dmask} -dshow {dshow} -suffix {suffix}')


# # GPT 复合函数
# seed_list = [1]
# target = 'composition'
# dir_suffix = '复合函数_不同层数和head数对结果的影响'
# lr = 2e-5
# batch_size = 100
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# model = 'GPT'
# N_train = 8000

# dname = ['13', '23', '43', '31', '32', '34'] + ['12', '14', '21', '41', '24', '42'] + ['11', '22', '33', '44']
# dmode = ['13', '23', '43', '31', '32', '34'] + ['12', '14', '21', '41', '24', '42'] + ['11', '22', '33', '44']
# dmask = [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0]
# dshow = [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0]
# dpercent = [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1]

# dn = ' '.join(map(str, dname))
# dp = ' '.join(map(str, dpercent))
# dmode = ' '.join(map(str, dmode))
# dmask = ' '.join(map(str, dmask))
# dshow = ' '.join(map(str, dshow))

# suffix = '2层_1head'

# os.system(f'/bin/python -m main -N_train {N_train} -seed 1 -func {target} -lr {lr} -m {model}\
#                 -scheduler {scheduler} -ne 8000 -nl 2 -nh 1 -bs {batch_size} -dir_suffix {dir_suffix} \
#                 -dmode {dmode} -dp {dp} -dn {dn} -dmask {dmask} -dshow {dshow} -suffix {suffix}')


# # GPT 3x_to_x 4层
# # seed_list = [1,2,3,4,5,6,7,8,9,10]
# seed_list = [2]
# # seed_list = [3,4,5,6]
# target = '3x_to_x'
# target = '3x_to_x_new_interval'
# # target = 'x3_to_x_new_interval'
# target = '3x_to_x_round'
# lr = 2e-5
# # model='GPT_lightly'
# model='GPT'
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# batch_size = 50
# N_train_list = [1000]
# dir_suffix = '2层_1head_3x_to_x'
# suffix = '2Decoder_3x间隔2位'
# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -m {model} -func {target} -lr {lr} \
#                   -scheduler {scheduler} -ne 4000 -nl 2 -nh 1 -bs {batch_size} -dir_suffix {dir_suffix} -suffix {suffix}\
#                     -dm 400 -dk 64 -dv 64\
#                     --use_mod_list_specific True --data_distance 2')
#                 #   -sme 10 -ple 1 -pae 1 -plae 50')

# # GPT 3x_to_x 4层
# # seed_list = [1,2,3,4,5,6,7,8,9,10]
# seed_list = [1]
# # seed_list = [3,4,5,6]
# target = '3x_to_x_1pos'
# target = '2task'
# lr = 2e-5
# model='GPT'
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# batch_size = 50
# N_train_list = [1000]
# dir_suffix = 'test_1_pos'
# suffix = '2prompt_qkv3'
# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -m {model} -func {target} -lr {lr} \
#                   -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} -suffix {suffix}\
#                     -dm 400 -dk 64 -dv 64')


# # GPT 3x_to_x 4x_to_x 奇偶区分 4层
# # seed_list = [1,2,3,4,5,6,7,8,9,10]
# seed_list = [1]
# # seed_list = [3,4,5,6]
# target = '34x_to_x_odd_even'
# lr = 2e-5
# model='GPT'
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# batch_size = 50
# N_train_list = [1000, 2000, 3000]
# dir_suffix = '3x_to_x_4x_to_x_odd_even'
# suffix = '4layer_4head'

# dname = ['promt_3_x_odd', 'promt_3_x_even', 'promt_4_x_odd', 'promt_4_x_even']
# dmode = ['3x_odd', '3x_even', '4x_odd', '4x_even']
# dmask = [0, 1, 1, 0]
# dshow = [1, 1, 1, 1]
# dpercent = [1, 1, 1, 1]

# dn = ' '.join(map(str, dname))
# dp = ' '.join(map(str, dpercent))
# dmode = ' '.join(map(str, dmode))
# dmask = ' '.join(map(str, dmask))
# dshow = ' '.join(map(str, dshow))

# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -func {target} -lr {lr} -m {model}\
#                         -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} \
#                         -dmode {dmode} -dp {dp} -dn {dn} -dmask {dmask} -dshow {dshow} -suffix {suffix}')



# # GPT 3x_to_x 4x_to_x+1 区间区分 4层
# # seed_list = [1,2,3,4,5,6,7,8,9,10]
# seed_list = [1]
# # seed_list = [3,4,5,6]
# target = '3x_to_x_4x_to_x+1_two_interval'
# lr = 2e-5
# model='GPT'
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# batch_size = 50
# N_train_list = [10000]
# dir_suffix = '3x_to_x_4x_to_x+1_two_interval'
# suffix = '4layer_4head_区间20-60_区间60-100'

# dname = ['promt_3_x_I1', 'promt_3_x_I2', 'promt_4_x+1_I1', 'promt_4_x+1_I2']
# dmode = ['3x_I1', '3x_I2', '4x+1_I1', '4x+1_I2']
# dmask = [0, 1, 1, 0]
# dshow = [1, 1, 1, 1]
# dpercent = [1, 1, 1, 1]

# dn = ' '.join(map(str, dname))
# dp = ' '.join(map(str, dpercent))
# dmode = ' '.join(map(str, dmode))
# dmask = ' '.join(map(str, dmask))
# dshow = ' '.join(map(str, dshow))

# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -func {target} -lr {lr} -m {model}\
#                         -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} \
#                         -dmode {dmode} -dp {dp} -dn {dn} -dmask {dmask} -dshow {dshow} -suffix {suffix}')



# # GPT 3x_to_x 4x_to_x+1 5x_to_x+2 区间区分 4层
# # seed_list = [1,2,3,4,5,6,7,8,9,10]
# seed_list = [1]
# # seed_list = [3,4,5,6]
# target = '345x_three_interval'
# lr = 2e-5
# model='GPT'
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# batch_size = 50
# N_train_list = [18000]
# dir_suffix = '345x_three_interval'
# suffix = '4L1H_区间20-70_区间20-100_区间50-100'

# dname = ['promt_3_x_I1', 'promt_3_x_I2', 'promt_3_x_I3', 'promt_4_x+1_I1', 'promt_4_x+1_I2', 'promt_4_x+1_I3', 'promt_5_x+2_I1', 'promt_5_x+2_I2', 'promt_5_x+2_I3']
# dmode = ['3x_I1', '3x_I2', '3x_I3', '4x+1_I1', '4x+1_I2', '4x+1_I3', '5x+2_I1', '5x+2_I2', '5x+2_I3']
# dmask = [0, 1, 1] + [1, 0, 1] + [1, 1, 0]
# dshow = [1, 1, 1] + [1, 1, 1] + [1, 1, 1]
# dpercent = [1, 1, 1] + [1, 1, 1] + [1, 1, 1]

# dn = ' '.join(map(str, dname))
# dp = ' '.join(map(str, dpercent))
# dmode = ' '.join(map(str, dmode))
# dmask = ' '.join(map(str, dmask))
# dshow = ' '.join(map(str, dshow))

# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -func {target} -lr {lr} -m {model}\
#                         -scheduler {scheduler} -ne 4000 -nl 4 -nh 1 -bs {batch_size} -dir_suffix {dir_suffix} \
#                         -dmode {dmode} -dp {dp} -dn {dn} -dmask {dmask} -dshow {dshow} -suffix {suffix}\
#                         --use_mod_list_specific True')

# GPT multianchor_multiinterval
# seed_list = [1]
# target = 'multianchor_multiinterval'
# lr = 2e-5
# model='GPT'
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# batch_size = 50
# N_train_list = [20000]
# dir_suffix = '每个anchor有各自的区间'
# suffix = '4L4H'
# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -N_test {2000} -seed {seed} -m {model} -func {target} -lr {lr} \
#                   -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} -suffix {suffix}\
#                     -dm 400 -dk 64 -dv 64 --use_mod_list_specific True -sme 10')


# # GPT 3x_to_x 概率输出 4层
# # seed_list = [1,2,3,4,5,6,7,8,9,10]
# seed_list = [1]
# # seed_list = [3,4,5,6]
# target = '3x_to_x_probability'
# lr = 2e-5
# model='GPT'
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# batch_size = 50
# N_train_list = [10000]
# dir_suffix = '3x_to_x_or_x+1_依概率输出'

# dname = ['3x_to_x_x+1_probality']
# dmode = ['probability']
# dmask = [0]
# dshow = [1]
# dpercent = [1]

# dn = ' '.join(map(str, dname))
# dp = ' '.join(map(str, dpercent))
# dmode = ' '.join(map(str, dmode))
# dmask = ' '.join(map(str, dmask))
# dshow = ' '.join(map(str, dshow))

# # for p in [0.2, 0.4, 0.5, 0.6, 0.8]:
# for p in [0.8]:
#     suffix = f'{int(100*p)}概率x_{int(100*(1-p))}概率x+1_大数据小epoch'
#     for seed in seed_list:
#         for N_train in N_train_list:
#             os.system(f'/bin/python -m main -N_train {N_train} -N_test 100 -seed {seed} -func {target} -lr {lr} -m {model}\
#                 -scheduler {scheduler} -ne 100 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} \
#                 -dmode {dmode} -dp {dp} -dn {dn} -dmask {dmask} -dshow {dshow} -suffix {suffix} --3x_to_x_probability {p}\
#                 -sme 1 -ple 1 -pae 1 -plae 20')


# # GPT 依概率输出 概率与距离成反比
# seed_list = [1]
# # seed_list = [3,4,5,6]
# target = '3x_to_x_distance_related'
# lr = 2e-5
# model='GPT'
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# batch_size = 50
# N_train_list = [30000]
# dir_suffix = '3x_to_x_x选择与距离呈反比_循环定义'
# suffix = f'不区分训练集测试集'

# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -func {target} -lr {lr} -m {model}\
#             -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} \
#             -suffix {suffix}')


# # GPT 3x1x2_to_x1+x2
# # seed_list = [1,2,3,4,5,6,7,8,9,10]
# seed_list = [1]
# # seed_list = [3,4,5,6]
# target = '3x1x2_to_x1+x2'
# lr = 2e-5
# model='GPT'
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# batch_size = 100
# N_train_list = [2000]
# dir_suffix = '3x1x2_to_x1+x2'
# suffix = '2层_1head'
# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -m {model} -func {target} -lr {lr} \
#                   -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} -suffix {suffix}\
#                     -dm 400 -dk 64 -dv 64')
#                 #   -sme 10 -ple 1 -pae 1 -plae 50')



# # GPT 上下文学习
# seed_list = [1]
# target = 'context'
# lr = 2e-5
# N_train_list = [500, 1000, 2000, 3000, 4000]
# batch_size = 100
# dp = [0.5, 0.5]
# dmode = ['abab', 'abba']
# dn = ['forward', 'reverse']
# dmask = [0, 0]
# dshow = [1, 1]

# dn = ' '.join(map(str, dn))
# dp = ' '.join(map(str, dp))
# dmode = ' '.join(map(str, dmode))
# dmask = ' '.join(map(str, dmask))
# dshow = ' '.join(map(str, dshow))

# model = 'GPT_new'
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# dir_suffix = f'{target}_for_paper_no_mask'
# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -func {target} -lr {lr} -m {model}\
#                     -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} \
#                     -dmode {dmode} -dp {dp} -dn {dn} -dmask {dmask} -dshow {dshow}')



# # GPT 验证频率原则
# seed_list = [1]
# target = '3x_to_x_4x_to_x+1'
# lr = 2e-5
# model='GPT'
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# batch_size = 50
# dp = [0.5, 0.5]
# dmode = ['3x_to_x', '4x_to_x+1']
# dn = ['3x_to_x', '4x_to_x+1']
# dmask = [0, 0]
# dshow = [1, 1]

# dn = ' '.join(map(str, dn))
# dp = ' '.join(map(str, dp))
# dmode = ' '.join(map(str, dmode))
# dmask = ' '.join(map(str, dmask))
# dshow = ' '.join(map(str, dshow))
# N_train_list = [1000]
# dir_suffix = '验证频率原则'
# suffix = '4L4H'
# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -N_test {1000} -seed {seed} -m {model} -func {target} -lr {lr} \
#                   -scheduler {scheduler} -ne 300 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} -suffix {suffix}\
#                     -dm 400 -dk 64 -dv 64 --use_mod_list_specific True -sme 1 -ple 1 -pae 1 -plae 10\
#                     -dmode {dmode} -dp {dp} -dn {dn} -dmask {dmask} -dshow {dshow}')





# # DNN 输出第5位
# target = 'output_5th_word'
# seed_list = [1]
# lr = 2e-5
# model='DNN'
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# batch_size = 100
# N_train_list = [1000]
# dir_suffix = '输出第5位'
# suffix = '2L400d'
# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -m {model} -func {target} -lr {lr} \
#                   -scheduler {scheduler} -ne 4000 -nl 2 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} -suffix {suffix}\
#                     -dm 400 -dk 64 -dv 64')
#                 #   -sme 10 -ple 1 -pae 1 -plae 50')


# GPT 输出3和4最中间的那个词
# seed_list = [1,2,3,4,5,6,7,8,9,10]
seed_list = [1]
# seed_list = [3,4,5,6]
target = 'output_middle_word'
lr = 2e-5
model='GPT'
scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
batch_size = 50
N_train_list = [30000]
dir_suffix = '输出3和4最中间的那个词'
suffix = '4L4H'

dname = ['3x4', '3xxx4', '3xxxxx4']
dmode = ['3x4', '3xxx4', '3xxxxx4']
dmask = [0, 0, 0]
dshow = [1, 1, 1]
dpercent = [1, 1, 1]

dn = ' '.join(map(str, dname))
dp = ' '.join(map(str, dpercent))
dmode = ' '.join(map(str, dmode))
dmask = ' '.join(map(str, dmask))
dshow = ' '.join(map(str, dshow))

for seed in seed_list:
    for N_train in N_train_list:
        os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -func {target} -lr {lr} -m {model}\
                        -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} \
                        -dmode {dmode} -dp {dp} -dn {dn} -dmask {dmask} -dshow {dshow} -suffix {suffix}\
                        --use_mod_list_specific True')