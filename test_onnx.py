import torch
import onnxruntime
import numpy as np
import archs_ucm
import argparse
import yaml
import os

# Wrapper for PyTorch model to ensure inference_mode=True is used
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_tensor):
        return self.model(input_tensor, inference_mode=True)

def main():
    parser = argparse.ArgumentParser(description="Test ONNX model consistency with PyTorch model")
    parser.add_argument('--pytorch_model_path', type=str, required=True,
                        help='Path to the trained PyTorch model (.pth file)')
    parser.add_argument('--onnx_model_path', type=str, required=True,
                        help='Path to the exported ONNX model (.onnx file)')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the config.yml file for model configuration')

    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load PyTorch model
    print(f"Loading PyTorch model: {config['arch']}")
    pytorch_model = archs_ucm.__dict__[config['arch']](
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        embed_dims=config.get('embed_dims', [64, 128, 256, 512]),
        num_heads=config.get('num_heads', [1, 2, 4, 8]),
        mlp_ratios=config.get('mlp_ratios', [4, 4, 4, 4]),
        qkv_bias=config.get('qkv_bias', False),
        qk_scale=config.get('qk_scale', None),
        drop_rate=config.get('drop_rate', 0.),
        attn_drop_rate=config.get('attn_drop_rate', 0.),
        drop_path_rate=config.get('drop_path_rate', 0.),
        norm_layer_name=config.get('norm_layer', 'LayerNorm'),
        depths=config.get('depths', [1, 1, 1]),
        sr_ratios=config.get('sr_ratios', [8, 4, 2, 1]),
        img_size=config.get('input_h', 256)
    )

    print(f"Loading PyTorch model weights from: {args.pytorch_model_path}")
    pytorch_model.load_state_dict(torch.load(args.pytorch_model_path, map_location=device))
    pytorch_model.to(device)
    pytorch_model.eval()

    # Wrap PyTorch model for consistent forward pass signature
    wrapped_pytorch_model = ModelWrapper(pytorch_model)
    wrapped_pytorch_model.to(device)
    wrapped_pytorch_model.eval()

    # Load ONNX model
    print(f"Loading ONNX model from: {args.onnx_model_path}")
    ort_session = onnxruntime.InferenceSession(args.onnx_model_path, providers=['CPUExecutionProvider']) # Or ['CUDAExecutionProvider'] if testing on GPU

    # Create dummy input
    batch_size = 1
    input_channels = config['input_channels']
    input_h = config['input_h']
    input_w = config['input_w']

    # For PyTorch
    dummy_input_pytorch = torch.randn(batch_size, input_channels, input_h, input_w, requires_grad=False).to(device)

    # For ONNX (needs numpy array)
    dummy_input_onnx = dummy_input_pytorch.cpu().numpy().astype(np.float32)

    print(f"Created dummy input with shape: {dummy_input_pytorch.shape}")

    # PyTorch model inference
    print("Running PyTorch model inference...")
    with torch.no_grad():
        pytorch_output = wrapped_pytorch_model(dummy_input_pytorch)

    # Handle tuple output from PyTorch model if deep_supervision was on during export (it is not for inference_mode=True)
    if isinstance(pytorch_output, tuple):
        pytorch_output_np = pytorch_output[0].cpu().numpy() # Assuming the main output is the first element
    else:
        pytorch_output_np = pytorch_output.cpu().numpy()

    # ONNX model inference
    print("Running ONNX model inference...")
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    onnx_output = ort_session.run([output_name], {input_name: dummy_input_onnx})[0]

    # Compare outputs
    print("\n--- Output Comparison ---")
    print(f"PyTorch output shape: {pytorch_output_np.shape}")
    print(f"ONNX output shape: {onnx_output.shape}")

    shapes_match = pytorch_output_np.shape == onnx_output.shape
    print(f"Shapes match: {shapes_match}")

    if shapes_match:
        abs_diff = np.abs(pytorch_output_np - onnx_output)
        max_abs_diff = np.max(abs_diff)
        all_close = np.allclose(pytorch_output_np, onnx_output, rtol=1e-03, atol=1e-05)
        print(f"Outputs numerically close (allclose check): {all_close}")
        print(f"Maximum absolute difference: {max_abs_diff:.6e}")
        if not all_close:
            print("Differences detected. Consider adjusting tolerances or investigating model discrepancies.")
    else:
        print("Cannot perform numerical comparison because shapes do not match.")

if __name__ == '__main__':
    main()
