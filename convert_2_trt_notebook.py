#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
import torch.onnx
import os
import torch2trt
import onnx
import tensorrt as trt


# In[11]:


def load_resnet18(model_path):
    custom_resnet18 = models.resnet18(num_classes=10)
    custom_resnet18.load_state_dict(torch.load(model_path))
    return custom_resnet18


# In[23]:


example_input = torch.randn(1,3, 96, 96)
print()
#model_path = os.getcwd() + '/data/experiments/train_resnet18_zero_shot_train_only/checkpoint_28.pth'
onnx_path = "resnet18.onnx"  # Save the converted ONNX model
engine_path = "resnet18.trt"  # Save the TensorRT engine

# Load the ResNet18 model
#custom_resnet18 = load_resnet18(model_path)

# Set the model to evaluation mode
#custom_resnet18.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 96, 96)

# Export the model to ONNX
#torch.onnx.export(custom_resnet18, dummy_input, onnx_path, verbose=True)
onnx_model = onnx.load(onnx_path)


# In[24]:


USE_FP16 = True
if USE_FP16:
    get_ipython().system('trtexec --onnx=onnx_model --saveEngine=resnet_engine_pytorch.trt  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16')
else:
    get_ipython().system('trtexec --onnx=onnx_model --saveEngine=resnet_engine_pytorch.trt  --explicitBatch')


# In[5]:


trt.__version__


# In[ ]:




