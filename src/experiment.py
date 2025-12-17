"""
Experiment tracking module for versioned checkpoints and run logging.
"""
import os
import re
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def _convert_to_native(obj):
    """Recursively convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_native(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class ExperimentManager:
    """
    Manages experiment checkpoints with auto-incrementing run IDs and JSON logging.
    
    Usage:
        exp = ExperimentManager(checkpoint_dir='checkpoints')
        run_id = exp.save_run(model, model_type='mlp', config={...}, results={...})
    """
    
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.checkpoint_dir / 'experiments.json'
    
    def _get_next_run_number(self, model_type: str) -> int:
        """Scan checkpoint directory to find the next available run number."""
        pattern = re.compile(rf'^{model_type}_run_(\d+)\.pt$')
        max_num = 0
        
        for f in self.checkpoint_dir.iterdir():
            match = pattern.match(f.name)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
        
        return max_num + 1
    
    def _load_log(self) -> list:
        """Load existing experiments log or return empty list."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    content = f.read()
                    if not content.strip():
                        return []
                    return json.loads(content)
            except json.JSONDecodeError as e:
                # Backup corrupted file and start fresh
                backup_path = self.log_file.with_suffix('.json.corrupted')
                self.log_file.rename(backup_path)
                print(f"WARNING: experiments.json was corrupted, backed up to {backup_path}")
                return []
        return []
    
    def _save_log(self, entries: list):
        """Save experiments log to JSON file using atomic write to prevent corruption."""
        # Convert NumPy types to native Python types for JSON serialization
        entries = _convert_to_native(entries)
        
        # Write to temp file first, then atomically rename to prevent corruption
        temp_file = self.log_file.with_suffix('.json.tmp')
        with open(temp_file, 'w') as f:
            json.dump(entries, f, indent=2)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk
        
        # Atomic rename (on POSIX systems, this is atomic)
        temp_file.replace(self.log_file)
    
    def save_run(self, model, model_type: str, config: dict, results: dict) -> str:
        """
        Save a model checkpoint with versioned naming and log the run.
        
        Args:
            model: PyTorch model to save
            model_type: 'mlp' or 'cnn'
            config: Hyperparameters dict
            results: Training results dict (final_rank, key_recovered, etc.)
        
        Returns:
            run_id: The assigned run ID (e.g., 'mlp_run_001')
        """
        # Get next run number
        run_num = self._get_next_run_number(model_type)
        run_id = f"{model_type}_run_{run_num:03d}"
        
        # Build checkpoint filename
        checkpoint_filename = f"{run_id}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        # Build checkpoint dict (include model-specific attributes)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'metadata': results,
            'run_id': run_id,
        }
        
        # Add model-specific attributes
        if hasattr(model, 'input_dim'):
            checkpoint['input_dim'] = model.input_dim
        if hasattr(model, 'shared_dim'):
            checkpoint['shared_dim'] = model.shared_dim
        if hasattr(model, 'input_size'):
            checkpoint['input_size'] = model.input_size
        if hasattr(model, 'filters'):
            checkpoint['filters'] = model.filters
        if hasattr(model, 'flat_size'):
            checkpoint['flat_size'] = model.flat_size
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Create log entry
        timestamp = datetime.now().isoformat(timespec='seconds')
        log_entry = {
            'run_id': run_id,
            'timestamp': timestamp,
            'model_type': model_type,
            'checkpoint_path': str(checkpoint_path),
            'config': config,
            'results': results,
        }
        
        # Append to log
        entries = self._load_log()
        entries.append(log_entry)
        self._save_log(entries)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        print(f"Run logged: {run_id} @ {timestamp}")
        
        return run_id
    
    def list_runs(self, model_type: str = None) -> list:
        """
        List all logged runs, optionally filtered by model type.
        
        Args:
            model_type: Filter by 'mlp' or 'cnn', or None for all
        
        Returns:
            List of run entries
        """
        entries = self._load_log()
        
        if model_type:
            entries = [e for e in entries if e['model_type'] == model_type]
        
        return entries
    
    def print_runs(self, model_type: str = None):
        """Print a summary table of all runs."""
        entries = self.list_runs(model_type)
        
        if not entries:
            print("No runs found.")
            return
        
        print(f"{'Run ID':<16} {'Timestamp':<20} {'Type':<6} {'Rank':<6} {'Recovered'}")
        print("-" * 70)
        
        for e in entries:
            run_id = e['run_id']
            ts = e['timestamp']
            mtype = e['model_type']
            rank = e['results'].get('final_rank', 'N/A')
            recovered = e['results'].get('key_recovered', 'N/A')
            print(f"{run_id:<16} {ts:<20} {mtype:<6} {rank:<6} {recovered}")
    
    def load_run(self, run_id: str, device: str = 'cpu'):
        """
        Load a checkpoint by run ID.
        
        Args:
            run_id: The run ID to load (e.g., 'mlp_run_001')
            device: Device to load model to
        
        Returns:
            Tuple of (model, config, results) or raises FileNotFoundError
        """
        # Find the entry in the log
        entries = self._load_log()
        entry = next((e for e in entries if e['run_id'] == run_id), None)
        
        if entry is None:
            raise FileNotFoundError(f"Run '{run_id}' not found in experiments log")
        
        checkpoint_path = entry['checkpoint_path']
        model_type = entry['model_type']
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})
        results = checkpoint.get('metadata', {})
        
        # Reconstruct model based on type
        if model_type == 'mlp':
            from src.model import MultiOutputNet
            model = MultiOutputNet(
                input_dim=checkpoint.get('input_dim', config.get('input_dim', 700)),
                shared_dim=checkpoint.get('shared_dim', config.get('MLP_SHARED_DIM', 0)),
                branch_l1=config.get('MLP_BRANCH_LAYER_1', 20),
                branch_l2=config.get('MLP_BRANCH_LAYER_2', 10),
                output_classes=config.get('MLP_OUTPUT_CLASSES', 2)
            )
        elif model_type == 'cnn':
            from src.cnn_model import CNNMultiOutputNet
            model = CNNMultiOutputNet(
                input_size=checkpoint.get('input_size', config.get('CNN_INPUT_SIZE', 480)),
                filters=config.get('CNN_FILTERS', 4),
                kernel1=config.get('CNN_KERNEL_1', 32),
                kernel2=config.get('CNN_KERNEL_2', 16),
                pool1=config.get('CNN_POOL_1', 2),
                pool2=config.get('CNN_POOL_2', 4)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Loaded run '{run_id}' from {checkpoint_path}")
        
        return model, config, results


