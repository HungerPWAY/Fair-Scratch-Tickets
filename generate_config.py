#import yaml 
import sys
import ruamel.yaml
#Architecture
arch = "ResNet18"

#task
task = "dense"

#Dataset
data_path = "/home/tangpw/FairSurrogates/"
dataset = "CelebA"

#name
name = f"{arch.lower()}_{task.lower()}_{dataset.lower()}"

#Learning Rate Policy
optimizer = "adam"
lr = 0.01
lr_policy = "cosine_lr"
loss = "BCEWithLogitLoss"

#Network training config
epochs = 3
weight_decay = 0.0001
momentum = 0.9
batch_size = 128

#Sparsity
conv_type = "DenseConv"
bn_type = "LearnedBatchNorm"
init = "kaiming_normal"
mode = "fan_in"
nonlinearity = "relu"
prune_rate = 0.0

#Hardware Setup
workers = 4

yaml_str = """\
# Architecture
arch: ResNet18

#task
task: dense


# ===== Dataset ===== #
data: "/home/tangpw/FairSurrogates/"
set: CelebA

#name 
name: ResNet18_dense_celeba_kn

# ===== Learning Rate Policy ======== #
optimizer: sgd
lr: 0.01
lr_policy: cosine_lr
loss: BCEWithLogitsLoss

# ===== Network training config ===== #
epochs: 3
weight_decay: 0.0001
momentum: 0.9
batch_size: 128

# ===== Sparsity =========== #
conv_type: DenseConv
bn_type: LearnedBatchNorm
init: kaiming_normal
mode: fan_in
nonlinearity: relu
prune_rate: 0.0

# ===== Hardware setup ===== #
workers: 4
"""


yaml = ruamel.yaml.YAML()
code = yaml.load(yaml_str)


config =  dict(arch = arch,
    task = task,
        data = data_path, set = dataset,
            name = name,
            optimizer = optimizer, lr = lr, lr_policy = lr_policy, loss = loss,
                epochs = epochs, weight_decay = weight_decay, momentum = momentum, batch_size = batch_size,
                    conv_type = conv_type, bn_type = bn_type, init = init, mode = mode, nonlinearity = nonlinearity, prune_rate = prune_rate,
                        workers = workers)

code.update(config)

with open("my.yml", "w") as f:
    yaml.dump(data=code, stream = f)