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

input_path = sys.argv[1]
output_path = sys.argv[2]

trans = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = trans(Image.open(input_path).convert('RGB'))

model_input = img.unsqueeze(0)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


with open(output_path, 'w') as f:
    json.dump(to_numpy(model_input).flatten(), f, cls=NumpyEncoder)

