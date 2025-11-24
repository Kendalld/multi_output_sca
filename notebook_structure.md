# Jupyter Notebook Structure: MLP_MO_Attack.ipynb

## Block 1: Setup & Configuration
* Import `torch`, `h5py`, `numpy`, `matplotlib.pyplot`.
* Set device (CUDA/MPS if available, else CPU).
* Define Constants: `BATCH_SIZE = 1000` (adjust based on VRAM, not RAM), `LEARNING_RATE = 0.001`.

## Block 2: Memory-Efficient Data Loader
* Define class `ASCADDataset(torch.utils.data.Dataset)`.
* **Key Logic:**
    ```python
    def __getitem__(self, idx):
        # Read from disk only when requested
        # trace shape: (700,) -> will be batched to (batch_size, 700)
        trace = self.h5_file['traces'][idx]
        # plaintext shape: scalar -> will be batched to (batch_size,)
        plaintext = self.h5_file['metadata']['plaintext'][idx] 
        # Note: Adjust key names based on actual H5 structure (e.g., 'inputs' vs 'metadata')
        return torch.from_numpy(trace).float(), torch.tensor(plaintext).long()
    ```
* Instantiate `DataLoader` with `num_workers=2` (keep low to save RAM overhead).
* **DataLoader output shapes:** `(batch_size, 700)` for traces, `(batch_size,)` for plaintexts.

## Block 3: The Multi-Output Model
* Define class `MultiOutputMLP(nn.Module)`.
* **Shared Layer:** `nn.Sequential(nn.Linear(input_dim, 200), nn.ReLU())`. Input shape: `(batch_size, 700)`, Output shape: `(batch_size, 200)`.
* **Branches:** ```python
    self.branches = nn.ModuleList([
        nn.Sequential(
            nn.Linear(200, 20),    # Branch hidden layer 1
            nn.ReLU(),
            nn.Linear(20, 10),     # Branch hidden layer 2
            nn.ReLU(),
            nn.Linear(10, 2)       # Output 2 classes (LSB 0 or 1)
        ) for _ in range(256)
    ])
    ```
* **Forward Pass:** Loop through `self.branches` and stack outputs to return shape `(batch_size, 256, 2)`.

## Block 4: Training Utilities
* **AES S-box:** Define or import the AES S-box lookup table (see README.md for definition).
* **Label Generator:** A function `get_all_labels(plaintexts, sbox)` that generates labels for all 256 key hypotheses:
    ```python
    def get_all_labels(plaintexts, sbox):
        """
        Generate labels for all 256 key hypotheses.
        
        Args:
            plaintexts: (batch_size,) tensor of uint8 plaintext bytes
            sbox: (256,) numpy array or tensor containing AES S-box lookup table
        
        Returns:
            labels: (batch_size, 256) tensor of labels (0 or 1) - LSB of Sbox(plaintext XOR k)
        """
        batch_size = plaintexts.shape[0]
        labels = torch.zeros(batch_size, 256, dtype=torch.long)
        
        # Convert plaintexts to numpy for indexing if needed
        plaintexts_np = plaintexts.cpu().numpy() if isinstance(plaintexts, torch.Tensor) else plaintexts
        
        for k in range(256):
            # Compute: intermediate = Sbox[plaintext XOR k]
            intermediate = sbox[plaintexts_np ^ k]
            # Extract LSB: label = intermediate & 1
            labels[:, k] = torch.tensor(intermediate & 1, dtype=torch.long)
        
        return labels
    ```
* **Loss Function:** A loop or vectorized sum that compares the Model Output `(batch_size, 256, 2)` with the Label Tensor `(batch_size, 256)`:
    ```python
    # Model output shape: (batch_size, 256, 2)
    # Labels shape: (batch_size, 256)
    total_loss = 0
    for k in range(256):
        total_loss += criterion(model_output[:, k, :], labels[:, k])
    ```

## Block 5: The MVP Training Loop
* Initialize model, optimizer (Adam), and lists for `correct_key_acc` and `wrong_key_acc`.
* **Iterate Epochs:**
    1.  Forward pass: `output = model(traces)` → shape `(batch_size, 256, 2)`.
    2.  Generate labels: `labels = get_all_labels(plaintexts, AES_SBOX)` → shape `(batch_size, 256)`.
    3.  Compute Loss (sum over 256 heads): Compare `output[:, k, :]` with `labels[:, k]` for each `k`.
    4.  Backward pass & Step.
    5.  **Important:** Calculate accuracy for the *correct* key (known from dataset) vs the average of *incorrect* keys.
       * Model predictions: `preds = output.argmax(dim=2)` → shape `(batch_size, 256)`.
       * Correct key accuracy: Compare `preds[:, correct_key]` with `labels[:, correct_key]`.
       * Wrong key accuracy: Compare `preds[:, k]` with `labels[:, k]` for all `k != correct_key`.
    6.  Print distinct separation: "Epoch 1: Correct Key Acc: 0.51 | Best Wrong Key Acc: 0.50".

## Block 6: Visualization
* Use `matplotlib` to plot two lines:
    1.  Accuracy of the correct key branch (Red).
    2.  Accuracy of the best incorrect key branch (Blue).
* **Goal:** The Red line should separate from the Blue cluster after a few epochs.