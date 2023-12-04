#!/usr/bin/bash
#SBATCH --job-name=idefics-mqa-60
#SBATCH --output=idefics-mqa-60.out
#SBATCH --error=idefics-mqa-60.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=20:00:00

source ~/.bashrc
conda activate idefics
cd /home/naveensu/
python train-idefics-clevr.py --wandb idefics-60 --dir idefics-60 --ratio 0.6
