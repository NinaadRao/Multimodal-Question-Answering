#!/usr/bin/bash
#SBATCH --job-name=eval-clevr-50-2
#SBATCH --output=eval-clevr-50-2.out
#SBATCH --error=eval-clevr-50-2.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=10:00:00

source ~/.bashrc
conda activate idefics
cd /home/naveensu/
python eval-idefics-clevr.py -m /data/user_data/naveensu/idefics-50-2 -o idefics-50-2_clevr.csv
