import casvit.rcvit as rcvit
import torch
import torch.nn as nn



def load_partial_weights(model, pretrained_path):
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
    
    # # 4. Initialize new layers
    # for name, module in model.named_modules():
    #     if name not in pretrained:
    #         if isinstance(module, (nn.Conv2d, nn.Linear)):
    #             nn.init.kaiming_normal_(module.weight)
    #             if module.bias is not None:
    #                 nn.init.zeros_(module.bias)
    #         elif isinstance(module, nn.BatchNorm2d):
    #             nn.init.ones_(module.weight)
    #             nn.init.zeros_(module.bias)
    
    return model, filtered

def get_rcvit_extended(pretrained_path):
    # Usage
    model = rcvit.rcvit_t_modified(num_classes=64)
    model, filtered_weights = load_partial_weights(model, pretrained_path)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    loaded_params = sum(v.numel() for v in filtered_weights.values())
    print(f"Loaded {loaded_params/1e6:.1f}M/{total_params/1e6:.1f}M parameters")

    # # Freeze early layers
    # for name, param in model.named_parameters():
    #     if any([k in name for k in ['network.0', 'network.1']]):  # Safer check
    #         param.requires_grad = False

    return model

if __name__ == "__main__":

    from training.lightning_train_function import lightning_train

    model = get_rcvit_extended('src/casvit/CASVIT_t.pth')
    model.dist = False
    lightning_train('configs/default_training_config.json', model)