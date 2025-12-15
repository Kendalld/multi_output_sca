import torch
import torch.nn as nn


class CNNMultiOutputNet(nn.Module):
    
    def __init__(self, input_size=480, filters=4, kernel1=32, kernel2=16, pool1=2, pool2=4):
        super(CNNMultiOutputNet, self).__init__()
        self.input_size = input_size
        self.filters = filters
        
        # Block 1: Conv → BN → AvgPool → ReLU
        self.block1 = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=kernel1),
            nn.BatchNorm1d(filters),
            nn.AvgPool1d(kernel_size=pool1),
            nn.ReLU()
        )
        
        # Block 2: Conv → BN → AvgPool → ReLU
        self.block2 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_size=kernel2),
            nn.BatchNorm1d(filters),
            nn.AvgPool1d(kernel_size=pool2),
            nn.ReLU()
        )
        
        # Compute flattened size after conv blocks
        self.flat_size = self._get_flat_size(input_size, kernel1, pool1, kernel2, pool2, filters)
        
        # 256 branches: direct output layer (no hidden layers)
        self.branches = nn.ModuleList([
            nn.Linear(self.flat_size, 2) for _ in range(256)
        ])
        
        self._init_weights()
    
    def _get_flat_size(self, input_size, k1, p1, k2, p2, filters):
        # After conv1: input_size - k1 + 1
        # After pool1: // p1
        # After conv2: - k2 + 1
        # After pool2: // p2
        size = (input_size - k1 + 1) // p1
        size = (size - k2 + 1) // p2
        return size * filters
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, input_size) -> (B, 1, input_size)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        
        outs = [branch(x) for branch in self.branches]
        return torch.stack(outs, dim=1)  # (B, 256, 2)


def save_cnn_model(model, filepath, config=None, metadata=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_size': model.input_size,
        'filters': model.filters,
        'flat_size': model.flat_size,
    }
    if config is not None:
        checkpoint['config'] = config
    if metadata is not None:
        checkpoint['metadata'] = metadata
    torch.save(checkpoint, filepath)
    print(f"CNN model saved to {filepath}")


def load_cnn_model(filepath, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model = CNNMultiOutputNet(
        input_size=checkpoint.get('input_size', config.get('CNN_INPUT_SIZE', 480)),
        filters=config.get('CNN_FILTERS', 4),
        kernel1=config.get('CNN_KERNEL_1', 32),
        kernel2=config.get('CNN_KERNEL_2', 16),
        pool1=config.get('CNN_POOL_1', 2),
        pool2=config.get('CNN_POOL_2', 4)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    metadata = checkpoint.get('metadata', {})
    print(f"CNN model loaded from {filepath}")
    return model, config, metadata


