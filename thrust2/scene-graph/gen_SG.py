import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
import requests
import os
import sys
import json
import argparse
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'RelTR'))
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine
from models.transformer import Transformer
from models.reltr import RelTR


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Process some images.')
parser.add_argument('--save_path', type=str, help='Path to the file to save scene graph')
parser.add_argument('--metadata', type=str, help='Path to the metadata of split images.')
parser.add_argument('--im_dir', type=str, default='../../../data/datasets/GQA/images', help='Path to image directory')
parser.add_argument('--batchsize', type=int, default=4, help='Batchsize')
parser.add_argument('--save_interval', type=int, default=1, help='Save scenegraphs after how many batches')

# Parse the arguments
args = parser.parse_args()

print('Loaded args')

# save_dir = '/'.join(args.save_path.split('/')[:-1])
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Device:{DEVICE}')

CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

position_embedding = PositionEmbeddingSine(128, normalize=True)
backbone = Backbone('resnet50', False, False, False)
backbone = Joiner(backbone, position_embedding)
backbone.num_channels = 2048

transformer = Transformer(d_model=256, dropout=0.1, nhead=8,
                          dim_feedforward=2048,
                          num_encoder_layers=6,
                          num_decoder_layers=6,
                          normalize_before=False,
                          return_intermediate_dec=True).to(DEVICE)

model = RelTR(backbone, transformer, num_classes=151, num_rel_classes = 51,
              num_entities=100, num_triplets=200).to(DEVICE)
print('Initialized model')

# The checkpoint is pretrained on Visual Genome
ckpt = torch.hub.load_state_dict_from_url(
    url='https://cloud.tnt.uni-hannover.de/index.php/s/PB8xTKspKZF7fyK/download/checkpoint0149.pth',
    map_location=DEVICE, check_hash=True)
model.load_state_dict(ckpt['model'])
model.eval()
print('Loaded checkpoint')

# Some transformation functions
transform = T.Compose([
    T.Resize((800,800)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def grayscale_to_rgb(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
          (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(DEVICE)
    return b


# Get all the images
images = pd.read_csv(args.metadata)
print('Read metadata')
images = list(set(images['image']))
print('Total images:',len(images))

# batch images:
images = [images[i:i+args.batchsize] for i in range(0,len(images),args.batchsize)]

scene_graphs = dict()

finished_batch = 0
for batch in images:
    im = []

    for im_name in batch:
        img = Image.open(os.path.join(args.im_dir,im_name))
        img = grayscale_to_rgb(img)
        im.append(transform(img).unsqueeze(0))
        img.close()

    im_batch = torch.vstack(im).to(DEVICE)

    # Model outputs
    outputs = model(im_batch)

    probas = outputs['rel_logits'].softmax(-1)[:, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[:, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[:, :, :-1]
    keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                        probas_obj.max(-1).values > 0.3)).to(DEVICE)

    # outputs['sub_boxes'] = outputs['sub_boxes'].cpu()
    # print(outputs['sub_boxes'].device)
    # print(keep.device)
    # print(keep[0])
    # img = img.to(DEVICE)
    # im = im_batch[0,:,:,:]
    size_dim = [800,800]
    sub_bboxes_scaled = [rescale_bboxes(outputs['sub_boxes'][i, keep[i]], size_dim) for i in range(im_batch.shape[0])]
    obj_bboxes_scaled = [rescale_bboxes(outputs['obj_boxes'][i, keep[i]], size_dim) for i in range(im_batch.shape[0])]

    keep_queries = [torch.nonzero(keep[i], as_tuple=True)[0] for i in range(keep.shape[0])]

    for i in range(len(keep_queries)):
        im_name = batch[i]

        triplets = set()
        for idx in keep_queries[i]:
            triplets.add(CLASSES[probas_sub[i,idx].argmax()]+' '+REL_CLASSES[probas[i,idx].argmax()]+' '+CLASSES[probas_obj[i,idx].argmax()])

        scene_graphs[im_name] = list(triplets)

    finished_batch += 1
    print(f'Finished batch {finished_batch}/{len(images)}')

    if finished_batch%args.save_interval==0:
        with open(args.save_path, 'w+') as f:
            json.dump(scene_graphs, f)

with open(args.save_path, 'w+') as f:
    json.dump(scene_graphs, f)