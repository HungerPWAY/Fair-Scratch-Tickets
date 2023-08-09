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

### YAML Name Key

```
(u)uc -> (unscaled) unsigned constant
(u)sc -> (unscaled) signed constant
(u)pt -> (unscaled) pretrained init
(u)kn -> (unscaled) kaiming normal
```

### Example Run

```bash
python main.py --config configs/smallscale/conv4/conv4_usc_unsigned.yml \
               --multigpu 0 \
               --name example \
               --data <path/to/data-dir> \
               --prune-rate 0.5
```

