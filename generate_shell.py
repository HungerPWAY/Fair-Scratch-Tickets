mutligpu = 0
#prune_rate = [0.05, 0.1, 0.4, 0.8, 0.95]
prune_rate = 0.05
fair_type = "dp"
lam_fair = [0.0, 0.06, 0.12, 0.18, 0.24]
#lam_fair = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3, 0.33, 0.36, 0.42]
fair_regularization = ["logistic", "linear", "hinge"]
print_val_freq = 10
print_freq  = 100
#config = ["configs/celeba/ResNet18sparse_celeba_kn.yml", "configs/celeba/ResNet18sparse_celeba_unc.yml", "configs/celeba/ResNet18sparse_celeba_sc.yml", "configs/celeba/ResNet18sparse_celeba_xn.yml"]
config = "configs/celeba/ResNet18dense_celeba.yml"

#pruning_type = ["kernel"]
'''
num_repeat = 3
with open('./evaluate.sh', 'w+', encoding = 'utf-8') as f:
    f.write("#!/bin/bash\n")
    for _ in range(num_repeat):
        for i in config:
            for j in prune_rate:
                for k in lam_fair:
                    original_shell = f"python main.py --config {i} --multigpu {mutligpu} --prune-rate {j}  --fair-type {fair_type} --lam-fair {k} --fair-regularization {fair_regularization} --print-val-freq {print_val_freq} --print-freq {print_freq} --evaluate --pretrained runs/ResNet18sparse_celeba/ResNet18_sparse_celeba_{}/prune_rate=0.05_fair_type=logistic_lam_fair={}/checkpoints/model_best.pth\n"
                    f.write(original_shell)
'''

'''
with open('./evaluate.sh', 'w+', encoding = 'utf-8') as f:
    f.write("#!/bin/bash\n")
    for lam in lam_fair:
        for prune_type in pruning_type:
                original_shell = f"python main.py --config {config} --multigpu {mutligpu} --prune-rate {prune_rate}  --fair-type {fair_type} --lam-fair {lam} --fair-regularization {fair_regularization} --print-val-freq {print_val_freq} --print-freq {print_freq} --evaluate --pretrained runs/ResNet18sparse_celeba/ResNet18_sparse_celeba_{prune_type}/prune_rate=0.05_fair_type=logistic_lam_fair={lam}/checkpoints/model_best.pth --conv-type SubnetConv_{prune_type} --name ResNet18_sparse_celeba_{prune_type}\n"
                f.write(original_shell)
'''

with open('./run_dense_signed_constant.sh','w+', encoding = 'utf-8') as f:
    f.write("#!/bin/bash\n")
    for lam in lam_fair:
        for regular in fair_regularization:
                original_shell = f"python main.py --config configs/celeba/ResNet18dense_celeba.yml --multigpu 1 --prune-rate 1  --fair-type dp --lam-fair {lam} --fair-regularization {regular} --print-val-freq 10 --print-freq 100\n"
                f.write(original_shell)


