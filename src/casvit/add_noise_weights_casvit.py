import casvit.rcvit as rcvit
import torch
import torch.nn as nn

def add_noise_to_weights(model,noise_factor=0.1):
    """Adds random noise proportional to the average L2 norm of each layer."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:  # Only modify trainable parameters
                layer_norm = torch.norm(param, p=2) / param.numel()  # Average L2 norm
                noise =  noise_factor * layer_norm * torch.randn_like(param)  
                param.add_(noise)  # Add noise proportional to layer's L2 norm

def load_partial_weights(model, pretrained_path, noise_factor=0.05):
    pretrained = torch.load(pretrained_path)
    pretrained = pretrained["model"]
    model_dict = model.state_dict()
    
    # 1. Filter compatible weights
    filtered = {k: v for k, v in pretrained.items() 
                if k in model_dict and v.shape == model_dict[k].shape}
    
    # 2. Handle classifier head
    if 'head.weight' in pretrained:
        if model.head.weight.shape != pretrained['head.weight'].shape:
            print("Re-initializing classifier head")
            nn.init.xavier_normal_(model.head.weight)
            if model.head.bias is not None:
                nn.init.zeros_(model.head.bias)
    
    # 3. Load compatible weights
    model.load_state_dict(filtered, strict=False)

    # 4. Add noise to model weights
    add_noise_to_weights(model, noise_factor)
    
    return model, filtered

def get_rcvit_extended(pretrained_path, noise_factor=0.1):
    # Load model and partially load weights
    model = rcvit.rcvit_t_modified(num_classes=64)
    model, filtered_weights = load_partial_weights(model, pretrained_path, noise_factor)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    loaded_params = sum(v.numel() for v in filtered_weights.values())
    print(f"Loaded {loaded_params/1e6:.1f}M/{total_params/1e6:.1f}M parameters with {noise_factor*100:.1f}% noise")

    return model

if __name__ == "__main__":
    from training.lightning_train_function import lightning_train

    model = get_rcvit_extended('src/casvit/CASVIT_t.pth', noise_factor=0.05)
    model.dist = False
    lightning_train('configs/default_training_config.json', model)
