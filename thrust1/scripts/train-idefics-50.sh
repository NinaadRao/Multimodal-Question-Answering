#!/usr/bin/bash
#SBATCH --job-name=idefics-mqa-50
#SBATCH --output=idefics-mqa-50.out
#SBATCH --error=idefics-mqa-50.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=20:00:00

source ~/.bashrc
conda activate idefics
cd /home/naveensu/
python train-idefics-clevr.py --wandb idefics-50 --dir idefics-50 --ratio 0.5
