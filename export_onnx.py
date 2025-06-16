import torch
import archs_ucm
import argparse
import os
import yaml

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        return self.model(input, inference_mode=True)

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch UCM-Net model to ONNX")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained PyTorch model file (e.g., models/ph2_nb_testing_batch_8/model.pth)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path for the exported ONNX model (e.g., models/ph2_nb_testing_batch_8/model.onnx)')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the config.yml file for model configuration')

    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    print(f"Initializing model: {config['arch']}")
    model = archs_ucm.__dict__[config['arch']](num_classes=config['num_classes'],
                                               input_channels=config['input_channels']) # Add other params from config as needed

    # Load trained weights
    print(f"Loading trained weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Wrap model for inference_mode=True
    wrapped_model = ModelWrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()

    # Create dummy input
    batch_size = 1 # Typical for ONNX export
    input_channels = config['input_channels']
    input_h = config['input_h']
    input_w = config['input_w']
    dummy_input = torch.randn(batch_size, input_channels, input_h, input_w).to(device)
    print(f"Created dummy input with shape: {dummy_input.shape}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Export to ONNX
    print(f"Exporting model to ONNX: {args.output_path}")
    try:
        torch.onnx.export(wrapped_model,
                          dummy_input,
                          args.output_path,
                          export_params=True,
                          opset_version=12,
                          do_constant_folding=True,
                          training=torch.onnx.TrainingMode.EVAL, # Explicitly set training mode
                          input_names=['input'],
                          output_names=['output'],
                           dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                                         'output': {0: 'batch_size', 2: 'height', 3: 'width'}}) # Assuming output shape also depends on H, W
        print(f"Model successfully exported to {args.output_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")

if __name__ == '__main__':
    main()
