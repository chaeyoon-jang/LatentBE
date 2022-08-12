# LatentBE Code Implementation
This is a simple implementation of [Improving Ensemble Distillation With Weight Averaging and Diversifying Perturbation](https://arxiv.org/pdf/2206.15047.pdf). 

I used the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. 

## How to use
At root dir,
- Train Teacher Models  
- `sh train_teacher_1.sh`
- `sh train_teacher_2.sh`
- `sh train_teacher_3.sh`
- `sh train_teacher_4.sh`

- Train Baseline Model (Knowledge Distillation Model)
`sh train_kd.sh`

- Train LatentBE Model
`sh train_latentbe.sh`

- Train LatentBE + div Model
`sh train_latentBE_div.sh`

## File Structure
```
./src
├── main.py - main (execute)
├── evaluation.py - test 
├── cnn_be.py - cnn model with batch ensemble network
├── cnn.py - simple cnn model
├── train.py - train teacher models
├── train_KD.py - train knowledge distillation model
├── train_latentBE_div.py - train LatentBE
├── train_latentBE.py - train LatentBE + diversity perturbation
└── utils - utilities
    ├── metric.py - metric code 
    └── utils.py - others (set seed, num_workers, etc.)
```
## Referneces
[1] Main repository of paper (https://github.com/cs-giung/distill-latentbe)
