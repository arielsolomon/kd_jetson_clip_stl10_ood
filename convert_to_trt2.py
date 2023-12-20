import torch
import torchvision.models as models
import torch.onnx
import os
import torch2trt
import onnx
# import trt_pose.models

# Load the PyTorch model
model_path = os.getcwd()+'/data/experiments/train_resnet18_zero_shot_train_only/checkpoint_28.pth'
custom_resnet18 = models.resnet18(num_classes=10)  # Assuming the last layer is adapted to 10 classes
custom_resnet18.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
custom_resnet18.eval()

# Example input dimensions (adjust according to your model's input requirements)
example_input = torch.randn(1,3, 96, 96)

# Export the model to ONNX
onnx_path = 'checkpoint_16.onnx'
torch.onnx.export(custom_resnet18, example_input, onnx_path, verbose=False)

print(f"Model exported to {onnx_path}")

# Load the ONNX model
onnx_model = onnx.load('checkpoint_16.onnx')
# Convert the ONNX model to TensorRT
trt_engine = torch2trt.torch2trt(onnx_model,example_input.cuda())

# Save the TensorRT engine
with open('engine.trt', 'wb') as f:
    f.write(trt_engine.serialize())
