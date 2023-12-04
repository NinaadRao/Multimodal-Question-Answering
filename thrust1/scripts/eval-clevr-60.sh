#!/usr/bin/bash
#SBATCH --job-name=eval-clevr-60
#SBATCH --output=eval-clevr-60.out
#SBATCH --error=eval-clevr-60.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=10:00:00

source ~/.bashrc
conda activate idefics
cd /home/naveensu/
python eval-idefics-clevr.py -m /data/user_data/naveensu/idefics-60 -o idefics-60_clevr.csv
