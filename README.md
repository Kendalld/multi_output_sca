# PyTorch MVP: Multi-Output MLP for Side-Channel Analysis

This project is a **Minimum Viable Product (MVP)** implementation of the `MLP_MO` (Multi-Output Multilayer Perceptron) architecture described by Hoang et al. (2023) [cite_start][cite: 7].

It is designed to run in a **Jupyter Notebook** environment using **PyTorch**, specifically optimized to handle the `ascadv2-extracted.h5` dataset within a **25GB RAM** constraint.

## 1. Project Goal
[cite_start]The objective is to recover the AES-128 secret key from power traces in a non-profiled setting by training a single neural network to predict all 256 possible key bytes simultaneously[cite: 31, 45].

## 2. Dataset & Memory Constraints
**File:** `ascadv2-extracted.h5`
**Constraint:** Max 25GB RAM usage.

To satisfy memory constraints, this implementation uses a **Lazy Loading** strategy via a custom PyTorch `Dataset` class.
* **Do Not** load the entire dataset into NumPy arrays at the start.
* **Do** keep the HDF5 file open and read only the specific batch of traces (`N=1000`) required for the current training step during `__getitem__`.
* **Data Structure:**
    * [cite_start]**Input:** Power traces (700 samples)[cite: 143].
    * [cite_start]**Labeling:** Labels are generated *on-the-fly* inside the training loop using the plaintext metadata to save memory[cite: 44, 100].

### H5 File Structure
Expected structure in `ascadv2-extracted.h5`:
* **`traces`**: Shape `(N, 700)` - Power traces as float32 arrays
* **`metadata/plaintext`**: Shape `(N,)` - Plaintext bytes (uint8) for each trace
* **`metadata/key`**: Shape `(16,)` - Correct AES-128 key bytes (optional, for validation only)

**Note:** The exact H5 structure may vary. Verify the keys by inspecting the file:
```python
import h5py
with h5py.File('dataset/ascadv2-extracted.h5', 'r') as f:
    print(list(f.keys()))
    if 'metadata' in f:
        print(list(f['metadata'].keys()))
```

## 3. MVP Architecture: `MLP_MO`
The model is a Directed Acyclic Graph (DAG) constructed using `nn.ModuleList`.



### A. Network Topology
| Component | Specification | Source |
| :--- | :--- | :--- |
| **Input** | [cite_start]Size: 700 samples (Float32) | [cite: 143] |
| **Shared Layer** | [cite_start]`Linear(700, 200)` $\rightarrow$ `ReLU` | [cite: 144, 206] |
| **Branches** | 256 independent parallel branches (one for each key hypothesis). [cite_start]| [cite: 112] |
| **Branch Hidden** | [cite_start]`Linear(200, 20)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(20, 10)` $\rightarrow$ `ReLU` | [cite: 114] |
| **Branch Output** | [cite_start]`Linear(10, 2)` (Unnormalized logits for CrossEntropy) | [cite: 114, 150] |

[cite_start]*Note: The paper specifies the branch hidden structure as "20x10-Relu"[cite: 114]. For this MVP, we interpret this as two dense layers (size 20 and 10) before the final classification.*

### B. Loss Function: Multi-Loss
The network minimizes the sum of losses across all 256 outputs simultaneously:
$$\mathcal{L}_{total} = \sum_{k=0}^{255} \text{CrossEntropyLoss}(\text{pred}_k, \text{target}_k)$$
[cite_start][cite: 119]

### C. Data Flow & Tensor Shapes
Understanding the tensor dimensions at each stage is critical for implementation:

| Stage | Tensor Shape | Description |
| :--- | :--- | :--- |
| **Input** | `(batch_size, 700)` | Power traces (float32) |
| **Shared Layer Output** | `(batch_size, 200)` | Common feature representation |
| **Branch Input** | `(batch_size, 200)` | Shared output fed to each branch |
| **Branch Output** | `(batch_size, 2)` | Logits for each key hypothesis |
| **Model Output** | `(batch_size, 256, 2)` | Stacked outputs from all 256 branches |
| **Labels** | `(batch_size, 256)` | Ground truth LSB values (0 or 1) for each key hypothesis |
| **Plaintext Input** | `(batch_size,)` | Plaintext bytes (uint8) from dataset |

**Note:** The label tensor `(batch_size, 256)` contains the LSB of `Sbox(plaintext ⊕ k)` for each key hypothesis `k` in `[0, 255]`.

## 4. AES S-box Definition
The label generation requires the AES S-box lookup table. Use the standard AES S-box:

```python
# AES S-box (Substitution Box) - 256 byte lookup table
AES_SBOX = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
], dtype=np.uint8)
```

**Alternative:** You can also use `Crypto.Cipher.AES` from `pycryptodome` or implement the S-box lookup directly in the label generation function.

## 6. Implementation Checklist
Follow this step-by-step order when building the notebook:

1. **Verify H5 File Structure**
   - [ ] Inspect `ascadv2-extracted.h5` to confirm dataset keys (`traces`, `metadata/plaintext`, etc.)
   - [ ] Verify trace shape is `(N, 700)` and plaintext shape is `(N,)`
   - [ ] Check if correct key is available in metadata (for validation)

2. **Implement Data Loading**
   - [ ] Create `ASCADDataset` class with lazy loading in `__getitem__`
   - [ ] Test DataLoader with small batch to verify shapes: `(batch_size, 700)` and `(batch_size,)`
   - [ ] Ensure proper type casting (float32 for traces, long for plaintexts)

3. **Define AES S-box**
   - [ ] Add AES S-box lookup table (see Section 4)

4. **Implement Model Architecture**
   - [ ] Create `MultiOutputMLP` class with shared layer `Linear(700, 200) → ReLU`
   - [ ] Create 256 branches using `nn.ModuleList`, each with structure: `Linear(200, 20) → ReLU → Linear(20, 10) → ReLU → Linear(10, 2)`
   - [ ] Implement forward pass to return shape `(batch_size, 256, 2)`
   - [ ] Verify model output shapes with a test batch

5. **Implement Label Generation**
   - [ ] Create `get_all_labels(plaintexts, sbox)` function
   - [ ] Test function returns shape `(batch_size, 256)` with values 0 or 1
   - [ ] Verify label computation: `LSB(Sbox[plaintext ⊕ k])` for each key `k`

6. **Implement Training Loop**
   - [ ] Initialize model, optimizer (Adam), loss function (CrossEntropyLoss)
   - [ ] Implement forward pass, label generation, loss computation
   - [ ] Add backward pass and optimizer step
   - [ ] Add accuracy tracking for correct key vs wrong keys
   - [ ] Add epoch logging

7. **Add Visualization**
   - [ ] Plot correct key accuracy (red line) vs best wrong key accuracy (blue line)
   - [ ] Verify separation occurs after a few epochs

8. **Validation & Testing**
   - [ ] Run for 20-30 epochs
   - [ ] Verify correct key accuracy separates from wrong keys (>63% vs ~50%)
   - [ ] Check memory usage stays within 25GB limit

## 7. Hyperparameters
* **Framework:** PyTorch
* [cite_start]**Optimizer:** ADAM (Default settings)[cite: 127].
* [cite_start]**Batch Size:** 1000[cite: 153].
* [cite_start]**Initialization:** He Uniform[cite: 161].
* [cite_start]**Epochs:** ~20-30 (Attack success is often visible by epoch 10)[cite: 209, 219].

## 8. Verification & Visualization
The notebook includes a plotting utility to track the "Attack Success Rate":
1.  **Red Line:** Average accuracy of the branch corresponding to the **Correct Key** (fixed in the dataset).
2.  **Blue Line:** Average accuracy of the **Incorrect Key** branches.
3.  [cite_start]**Success:** The Red line should separate clearly from the Blue line (e.g., >63% vs ~50%)[cite: 210, 214].

## 9. File Structure
Expected directory structure:
```
multi_output_SCA/
├── dataset/
│   └── ascadv2-extracted.h5       # Place your dataset here
├── docs/
│   └── Efficient_Nonprofiled_Side-Channel_Attack_Using_Multi-Output_Classification_Neural_Network.pdf
├── src/                            # (Optional) For future modular code)
├── MLP_MO_Attack.ipynb            # Main Jupyter notebook (to be created)
├── README.md                       # This file
├── notebook_structure.md           # Detailed notebook block structure
└── agent.md                        # AI agent context and constraints
```

**Note:** Ensure the H5 file is placed in the `dataset/` directory before running the notebook.

## 10. Prerequisites
* `torch`
* `h5py`
* `matplotlib`
* `numpy`
* `jupyter`