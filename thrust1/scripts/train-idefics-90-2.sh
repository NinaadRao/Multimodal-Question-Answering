#!/usr/bin/bash
#SBATCH --job-name=idefics-mqa-90
#SBATCH --output=idefics-mqa-90.out
#SBATCH --error=idefics-mqa-90.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=20:00:00

source ~/.bashrc
conda activate idefics
cd /home/naveensu/
python train-idefics-clevr.py --wandb idefics-90 --dir idefics-90 --ratio 0.9 --model idefics-90