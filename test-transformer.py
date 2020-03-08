import os
import sys
import torch
import torch.nn
import argparse
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import torch.onnx
from networks.resnet import resnet50
import onnx
import onnxruntime
import json
import math

input_path = sys.argv[1]
output_path = sys.argv[2]

img = Image.open(input_path)
width, height = img.size

crop_size = 224

trans = transforms.Compose([
    #transforms.Resize(crop_size),
    transforms.CenterCrop(crop_size),
    #transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = trans(img.convert('RGB'))
print(img)
img.save(output_path)