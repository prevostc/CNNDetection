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


input_path = sys.argv[1]
model_path = sys.argv[2]

torch_model = resnet50(num_classes=1)
state_dict = torch.load(model_path, map_location='cpu')
torch_model.load_state_dict(state_dict['model'])

# set the model to inference mode
torch_model.eval()

trans = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = trans(Image.open(input_path).convert('RGB'))
model_input = img.unsqueeze(0)

torch_output = torch_model(model_input)

print('probability of being synthetic: {:.2f}%'.format(torch_output.sigmoid().item() * 100))

# export model
torch.onnx.export(torch_model,               # model being run
                  model_input,                         # model input (or a tuple for multiple inputs)
                  "cnndetection.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})

import onnx

onnx_model = onnx.load("cnndetection.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime

ort_session = onnxruntime.InferenceSession("cnndetection.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(model_input)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_output), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")