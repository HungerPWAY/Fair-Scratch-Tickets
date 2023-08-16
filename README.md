# Fair Scratch Tickets: Finding Fair Sparse Networks without Weight Training

by Pengwei Tang, Wei Yao, Zhicong Li, Yong Liu

URL: https://openaccess.thecvf.com/content/CVPR2023/papers/Tang_Fair_Scratch_Tickets_Finding_Fair_Sparse_Networks_Without_Weight_Training_CVPR_2023_paper.pdf

## Starting an Experiment 

We use config files located in the ```configs/``` folder to organize our experiments. The basic setup for any experiment is:

```bash
python main.py --config <path/to/config> <override-args>
```

Common example ```override-args``` include ```--multigpu=<gpu-ids seperated by commas, no spaces>``` to run on GPUs, and ```--prune-rate``` to set the prune rate, ```weights_remaining``` in our paper, for an experiment. Run ```python main --help``` for more details.




### Example Run
```bash
python main.py --config configs/celeba/ResNet18sparse_celeba_sc.yml \
                --multigpu 0 \
                --prune-rate 0.05 \
                --fair-type dp \
                --lam-fair 0.0 \
                --fair-regularization logistic \
                --print-val-freq 10 \
                --print-freq 100
```

```
python main_adv.py --config configs/celeba/z_sparse.yml \
      --multigpu 0 \
      --print-val-freq 20 \
      --print-freq 100 --adv \
      --init signed_constant \
      --name CelebA_Adv_ResNet18_sparse_sc_filter \
      --trainer train_new_adv \
      --lambd 0 \
      --prune-rate 0.001 \
      --conv-type SubnetConv_filter
```

## Citation
```
@InProceedings{Tang_2023_CVPR,
    author    = {Tang, Pengwei and Yao, Wei and Li, Zhicong and Liu, Yong},
    title     = {Fair Scratch Tickets: Finding Fair Sparse Networks Without Weight Training},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {24406-24416}
}
```

## Acknowledgement

Our codes are modified from [[What’s Hidden in a Randomly Weighted Neural Network]](https://github.com/allenai/hidden-networks).
