import numpy as np
import h5py
import torch

from .config import AES_Sbox

# Load traces and metadata from ASCAD HDF5 file
def load_ascad(file_path, num_traces, start_idx, target_byte):
    with h5py.File(file_path, 'r') as f:
        # Profiling_traces for the training set
        traces = f['Profiling_traces']['traces'][start_idx : start_idx + num_traces]
        metadata = f['Profiling_traces']['metadata'][start_idx : start_idx + num_traces]
        
        # Only interested in specific byte of plaintext
        plaintexts = metadata['plaintext'][:, target_byte]
        real_key = metadata['key'][0, target_byte]
        
    return traces, plaintexts, real_key


def load_ascad_attack(file_path, num_traces, start_idx, target_byte):

    with h5py.File(file_path, 'r') as f:
        traces = f['Attack_traces']['traces'][start_idx : start_idx + num_traces]
        metadata = f['Attack_traces']['metadata'][start_idx : start_idx + num_traces]

        plaintexts = metadata['plaintext'][:, target_byte]
        real_key = metadata['key'][0, target_byte]
        
    return traces, plaintexts, real_key


def normalize_traces(traces):

    traces = (traces - np.mean(traces, axis=0)) / np.std(traces, axis=0)
    return torch.tensor(traces, dtype=torch.float32)


def generate_labels(plaintexts):

    labels = np.zeros((len(plaintexts), 256), dtype=np.longlong)
    
    for k in range(256):
        val = AES_Sbox[plaintexts ^ k]
        labels[:, k] = val & 1  # LSB leakage model
        
    return torch.tensor(labels, dtype=torch.long)


def compute_key_scores(model, traces_tensor, labels_tensor, device='cpu', batch_size=100):

    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    model.eval()
    dataset = TensorDataset(traces_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    val_accum_loss = torch.zeros(256, device=device)
    val_count = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            
            for k in range(256):
                val_accum_loss[k] += nn.CrossEntropyLoss(reduction='sum')(out[:, k, :], target[:, k])
            
            val_count += data.size(0)
    
    avg_branch_losses = val_accum_loss / val_count
    # Convert loss to score: negate so that lower loss = higher score
    key_scores = -avg_branch_losses.cpu().numpy()
    
    return key_scores

# Rank keys by score and find the rank of the real key.
def rank_keys(key_scores, real_key):

    sorted_keys = np.argsort(key_scores)[::-1]  # Descending order
    rank_of_real = np.where(sorted_keys == real_key)[0][0] + 1  # 1-based ranking
    
    return sorted_keys, rank_of_real


