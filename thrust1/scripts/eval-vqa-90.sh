#!/usr/bin/bash
#SBATCH --job-name=eval-vqa-90
#SBATCH --output=eval-vqa-90.out
#SBATCH --error=eval-vqa-90.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=10:00:00

source ~/.bashrc
conda activate idefics
cd /home/naveensu/
python eval-idefics-vqa.py -m /data/user_data/naveensu/idefics-90 -o idefics-90_vqa.csv
