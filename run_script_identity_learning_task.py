import os

# Identity Learning Task
# seed_list = [1,2,3,4,5,6,7,8,9,10]
seed_list = [1]
target = '3x_to_x'
lr = 2e-5
model='GPT'
# model='DNN'
scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
batch_size = 50
N_train_list = [1000]
for seed in seed_list:
    for N_train in N_train_list:
        os.system(f'python3 -m main -N_train {N_train} \
                  -seed {seed} -m {model} -func {target} -lr {lr} \
                  -scheduler {scheduler} -ne 4000 \
                  -nl 4 -nh 4 -bs {batch_size} \
                  -dm 400 -dk 64 -dv 64')
