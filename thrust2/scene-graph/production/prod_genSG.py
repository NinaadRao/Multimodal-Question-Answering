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
sys.path.append(os.path.join(current_dir, '../RelTR'))
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine
from models.transformer import Transformer
from models.reltr import RelTR

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
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(DEVICE)
    return b

class SceneGraphGenerator:
    def __init__(self):
        self.CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
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

        self.REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                        'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                        'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                        'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                        'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                        'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

        self.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = self.get_model()

        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_model(self):
        position_embedding = PositionEmbeddingSine(128, normalize=True)
        backbone = Backbone('resnet50', False, False, False)
        backbone = Joiner(backbone, position_embedding)
        backbone.num_channels = 2048

        transformer = Transformer(d_model=256, dropout=0.1, nhead=8,
                                dim_feedforward=2048,
                                num_encoder_layers=6,
                                num_decoder_layers=6,
                                normalize_before=False,
                                return_intermediate_dec=True).to(self.DEVICE)

        model = RelTR(backbone, transformer, num_classes=151, num_rel_classes = 51,
                    num_entities=100, num_triplets=200).to(self.DEVICE)

        # The checkpoint is pretrained on Visual Genome
        ckpt = torch.hub.load_state_dict_from_url(
            url='https://cloud.tnt.uni-hannover.de/index.php/s/PB8xTKspKZF7fyK/download/checkpoint0149.pth',
            map_location=self.DEVICE, check_hash=True)
        model.load_state_dict(ckpt['model'])
        model.eval()

        return model


    def generate_SG(self,image_path):
        '''
        Input: image path
        Output: Verbalized SG
        '''
        im = Image.open(image_path)
        im = grayscale_to_rgb(im)
        img = self.transform(im).unsqueeze(0)

        # Model outputs
        outputs = self.model(img.to(self.DEVICE))

        probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
        probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
        probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
        keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                            probas_obj.max(-1).values > 0.3)).to(self.DEVICE)


        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

        keep_queries = torch.nonzero(keep, as_tuple=True)[0]

        triplets = set()
        for idx in keep_queries:
            triplets.add(self.CLASSES[probas_sub[idx].argmax()]+' '+self.REL_CLASSES[probas[idx].argmax()]+' '+self.CLASSES[probas_obj[idx].argmax()])

        scene_graph = '. '.join(list(triplets))

        im.close()

        return scene_graph