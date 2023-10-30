#!/usr/bin/bash
#SBATCH --job-name=clevr-llava-mqa
#SBATCH --output=clevr-llava-mqa.out
#SBATCH --error=clevr-llava-mqa.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem-per-gpu=48G
#SBATCH --time=15:00:00

source ~/.bashrc
conda activate llava
cd /home/naveensu/Multimodal-Question-Answering/thrust1/LLaVA
python -m llava.serve.prompteng --model-path liuhaotian/llava-v1.5-7b --image-file "https://llava-vl.github.io/static/images/view.jpg" --load-4bit --eval-file one-word-final

