import torch
import torchvision.models as models
import torch.onnx
import os
import torch2trt
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
model = onnx.load('model.onnx')
import os
import torch
import torchvision.models as models
import tensorrt as trt
import onnx

def load_resnet18(model_path):
    custom_resnet18 = models.resnet18(num_classes=10)
    custom_resnet18.load_state_dict(torch.load(model_path))
    return custom_resnet18

def convert_to_tensorrt(onnx_model_path, precision_mode='fp32'):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)

    if precision_mode == 'fp16' and builder.platform_has_fast_fp16:
        builder.fp16_mode = True
    elif precision_mode == 'int8' and builder.platform_has_fast_int8:
        builder.int8_mode = True

    with builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # Parse ONNX model
        onnx_model = onnx.load(onnx_model_path)
        success = parser.parse(onnx_model.SerializeToString())
        if not success:
            raise RuntimeError("Failed to parse ONNX model")

        # Set the precision mode
        if precision_mode == 'fp16' and builder.platform_has_fast_fp16:
            builder.set_fully_half_precision_mode(True)
        elif precision_mode == 'int8' and builder.platform_has_fast_int8:
            builder.set_int8_mode(True)

        # Build TensorRT engine
        with builder.build_cuda_engine(network) as engine:
            return engine

def save_engine(engine, engine_path):
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())

if __name__ == "__main__":
    model_path = os.getcwd() + '/data/experiments/train_resnet18_zero_shot_train_only/checkpoint_28.pth'
    onnx_path = "resnet18.onnx"  # Save the converted ONNX model
    engine_path = "resnet18.trt"  # Save the TensorRT engine

    # Load the ResNet18 model
    custom_resnet18 = load_resnet18(model_path)

    # Set the model to evaluation mode
    custom_resnet18.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, 96, 96)

    # Export the model to ONNX
    torch.onnx.export(custom_resnet18, dummy_input, onnx_path, verbose=True)

    # Convert to TensorRT engine
    trt_engine = convert_to_tensorrt(onnx_path)

    # Save the TensorRT engine
    save_engine(trt_engine, engine_path)
