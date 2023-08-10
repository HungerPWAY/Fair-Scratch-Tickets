# Fair Scratch Tickets: Finding Fair Sparse Networks without Weight Training

by Pengwei Tang, Wei Yao, Zhicong Li, Yong Liu

URL: https://openaccess.thecvf.com/content/CVPR2023/papers/Tang_Fair_Scratch_Tickets_Finding_Fair_Sparse_Networks_Without_Weight_Training_CVPR_2023_paper.pdf
<!-- ![alt text](images/teaser.png) -->





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
