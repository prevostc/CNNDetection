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

input_path = sys.argv[1]
model_path = sys.argv[2]


trans = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = trans(Image.open(input_path).convert('RGB'))

model_input = img.unsqueeze(0)

#print(model_input[0][0][0])

print(model_input.shape) # torch.Size([1, 3, 224, 224])
print(model_input.dtype) # torch.float32

ort_session = onnxruntime.InferenceSession("cnndetection.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

#print( np.reshape(to_numpy(model_input), (150528))[150500:])
# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(model_input)}
ort_outs = ort_session.run(None, ort_inputs)
res = ort_outs[0]
print(res)
res = 1/(1 + np.exp(-res))
print(res)
prob = res[0][0]
print(prob)

print('probability of being synthetic: {:.2f}%'.format(prob * 100))

