# MQA-with-Scene-Graphs
This projects experiments with the recent approaches to enhancing visual question answering using scene graphs
The following plan was followed:
1. IDEFIC model was evaluated on GQA dataset to obtain 49% accuracy
2. IDEFICs model was prompted with ground truth scene graph to observe accuracy jump of 69%. This proves that scene graphs add valuable information for the model to answer questions. 
3. Scene graph generation model was used to generate scene graphs and prompt with the same.

Here is a description of the files:
1. org_dataset.py: Transform (get the required subset, run some statistics about the split)
2. hf_dataset.py: Load (Push dataset to huggingface hub so that it can be easily pulled in subsequent runs - especially on GPU compute)
3. batch_prompts.py: Takes the image id, question and puts it into the format necessary for inferencing (Can be made with and without SG)
4. gen_SG.py: Calls functions from the [RelTR repo](https://github.com/yrcong/RelTR) to generate scene graphs
5. finetune.py: Finetunes IDEFICs
6. inference.py: Uses chosen model to predict the outputs of the questions given image and optionally the SG as well
7. test_accuracy.py: Test accuracy by comparing predicted answers (also parses the returned string to get answers) to answers from dataset
8. Experiments: this folder holds all the experiments conducted. This will be supplemented with a list of what each Experiment tested. In order to comply with GitHub's file size limitation only the question_ids used are part of this folder for now
9. Production: This is a cleaner version of inferencing using the models trained here. This folder contains functions to call the inference on either one data point or a set of data points.

How to set up the repo:
1. Download [GQA dataset images and scene graphs](https://cs.stanford.edu/people/dorarad/gqa/download.html) and store in Dataset/GQA
2. Clone [RelTR repo](https://github.com/yrcong/RelTR) into this main folder (scene-graphs)
3. Set the path to GQA images in all the files (since the compute we were using already contained it, the path is different currently)
4. Download the train and test image folders into production/data from our [custom HuggingFace dataset](https://huggingface.co/datasets/yujiaw2/capstoneMQA)
