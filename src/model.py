import torch
import torch.nn as nn


class MultiOutputNet(nn.Module):
    
    def __init__(self, input_dim, shared_dim, branch_l1, branch_l2, output_classes):
        super(MultiOutputNet, self).__init__()
        self.shared_dim = shared_dim
        self.input_dim = input_dim
        
        if shared_dim > 0:
            self.shared_layer = nn.Sequential(
                nn.Linear(input_dim, shared_dim),
                nn.ReLU()
            )
            branch_input = shared_dim
        else:
            self.shared_layer = nn.Identity()
            branch_input = input_dim
            
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(branch_input, branch_l1),
                nn.ReLU(),
                nn.Linear(branch_l1, branch_l2),
                nn.ReLU(),
                nn.Linear(branch_l2, output_classes)
            ) for _ in range(256)
        ])

    def forward(self, x):
        shared = self.shared_layer(x)
        
        outs = []
        for branch in self.branches:
            outs.append(branch(shared))
        
        # Stack to (Batch, 256, output_classes)
        return torch.stack(outs, dim=1)


def save_model(model, filepath, config=None, metadata=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': model.input_dim,
        'shared_dim': model.shared_dim,
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    # Get config from checkpoint or use defaults
    config = checkpoint.get('config', {})
    
    # Create model with saved architecture
    model = MultiOutputNet(
        input_dim=checkpoint.get('input_dim', config.get('input_dim', 700)),
        shared_dim=checkpoint.get('shared_dim', config.get('MLP_SHARED_DIM', 0)),
        branch_l1=config.get('MLP_BRANCH_LAYER_1', 20),
        branch_l2=config.get('MLP_BRANCH_LAYER_2', 10),
        output_classes=config.get('MLP_OUTPUT_CLASSES', 2)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    metadata = checkpoint.get('metadata', {})
    
    print(f"Model loaded from {filepath}")
    return model, config, metadata


