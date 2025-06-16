import torch
import archs_ucm
import yaml
import os

def main():
    config_path = 'models/ph2_nb_testing_batch_8/config.yml'
    model_path = 'models/ph2_nb_testing_batch_8/model.pth'

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize model
    print(f"Initializing model: {config['arch']} for dummy model creation")
    # Ensure all necessary arguments from config are passed to the model constructor
    model = archs_ucm.__dict__[config['arch']](
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        embed_dims=config.get('embed_dims', [64, 128, 256, 512]),  # Provide defaults if not in config
        num_heads=config.get('num_heads', [1, 2, 4, 8]),
        mlp_ratios=config.get('mlp_ratios', [4, 4, 4, 4]),
        qkv_bias=config.get('qkv_bias', False),
        qk_scale=config.get('qk_scale', None),
        drop_rate=config.get('drop_rate', 0.),
        attn_drop_rate=config.get('attn_drop_rate', 0.),
        drop_path_rate=config.get('drop_path_rate', 0.),
        norm_layer_name=config.get('norm_layer', 'LayerNorm'), # Assuming 'LayerNorm' maps to nn.LayerNorm or custom
        depths=config.get('depths', [1, 1, 1]), # Adjusted to match typical UCM_Net usage
        sr_ratios=config.get('sr_ratios', [8, 4, 2, 1]),
        img_size=config.get('input_h', 256) # Assuming img_size can be derived from input_h
    )

    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save model state_dict
    torch.save(model.state_dict(), model_path)
    print(f"Dummy model saved to {model_path}")

if __name__ == '__main__':
    main()
