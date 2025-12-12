# Multi-Output Neural Network for Side-Channel Analysis

A non-profiled side-channel attack implementation using a multi-output neural network to recover AES key bytes from power traces.

## Project Overview

This project implements a **non-profiled deep learning side-channel attack** based on the paper ["Efficient Non-profiled Side-Channel Attack Using Multi-Output Classification Neural Network"](Research/Efficient_Nonprofiled_Side-Channel_Attack_Using_Multi-Output_Classification_Neural_Network.pdf).

**Key Features:**
- Multi-output MLP architecture with 256 independent branches (one per key candidate)
- LSB (Least Significant Bit) leakage model: `L_k = Sbox[P ⊕ k] mod 2`
- Works on the ASCAD dataset (ATMega8515 AES implementation)
- No profiling phase required - trains and attacks simultaneously

**How it Works:**
1. For each key candidate `k ∈ [0, 255]`, a separate classification branch learns to predict the LSB of `Sbox[plaintext ⊕ k]`
2. The correct key produces consistent labels (matching the actual leakage), resulting in lower loss
3. After training, keys are ranked by their loss - the lowest loss indicates the correct key

## Repository Structure

```
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/                      # Source code package
│   ├── __init__.py
│   ├── config.py             # Hyperparameters and configuration
│   ├── model.py              # MultiOutputNet model definition
│   └── utils.py              # Data loading and utility functions
├── mlpmp_SCA.ipynb           # Main training notebook
├── demo/                     # Demo scripts
│   └── demo.ipynb            # Demonstration notebook
├── checkpoints/              # Saved model weights
├── results/                  # Generated attack results
├── datasets/                 # Dataset files (see below)
└── Research/                 # Reference papers
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/multi_output_SCA.git
cd multi_output_SCA
```

### 2. Create Virtual Environment

```bash
# Using venv (Python 3.10+)
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
.\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

Download the ASCAD dataset from the official repository:
- [ASCAD v1 Fixed Key](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key)

Place the `ASCAD.h5` file in `datasets/ASCADv1/`:
```
datasets/ASCADv1/ASCAD.h5
```

### 5. Download Pre-trained Model (Optional)

Download the pre-trained model from:
- **[Google Drive Link - TODO: Add after training]**

Place the model file in `checkpoints/`:
```
checkpoints/multi_output_sca.pt
```

## How to Run

### Option 1: Run the Demo (Recommended)

If you have the pre-trained model:

```bash
cd demo
jupyter notebook demo.ipynb
```

The demo will:
1. Load the pre-trained model
2. Run the attack on 1000 traces from the attack set
3. Display the key recovery results
4. Save plots and summary to `results/`

### Option 2: Train from Scratch

To train a new model:

```bash
jupyter notebook mlpmp_SCA.ipynb
```

Run all cells to:
1. Load and preprocess traces
2. Train the multi-output network
3. Visualize attack results
4. Save the trained model to `checkpoints/`

## Expected Output

### Training Output

```
Epoch 1/30: 100%|██████████| 100/100 [00:17<00:00, 5.74it/s]
Epoch 1 Avg Loss: 0.6969
Best Key: 0xe0, Real Key Rank: 0, Real Key Score: -0.6865
...
Epoch 30/30: 100%|██████████| 100/100 [00:17<00:00, 5.80it/s]
Epoch 30 Avg Loss: 0.6512
Best Key: 0xe0, Real Key Rank: 0, Real Key Score: -0.5544

==================================================
ATTACK SUMMARY
==================================================
Target Key Byte:  0xe0
Final Rank:       1
Final Score:      -0.5544
Result:           KEY RECOVERED!
==================================================
```

### Demo Output

The demo generates:
- `results/key_scores.png` - Bar chart of all 256 key scores
- `results/score_distribution.png` - Histogram of score distribution
- `results/attack_summary.txt` - Text summary of attack results

## Configuration

Hyperparameters are defined in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | 0.0005 | Adam optimizer learning rate |
| `SHARED_DIM` | 0 | Shared layer dimension (0 = no shared layer) |
| `BRANCH_LAYER_1` | 20 | First hidden layer size per branch |
| `BRANCH_LAYER_2` | 10 | Second hidden layer size per branch |
| `NUM_EPOCHS` | 30 | Number of training epochs |
| `NUM_TRACES` | 10000 | Number of traces for training |
| `BATCH_SIZE` | 100 | Training batch size |
| `TARGET_BYTE_IDX` | 2 | Target key byte index (0-15) |

**Tuning Tips:**
- `SHARED_DIM = 200` reduces compute by sharing features across branches
- Increase `NUM_TRACES` for better accuracy (up to 50k for ASCAD)
- Key recovery typically occurs within ~6 epochs

## Model Architecture

```
Input (700 features)
       │
       ▼
[Optional Shared Layer]
  Linear(700 → SHARED_DIM) + ReLU
       │
       ├──────────────┬──────────────┬─── ... ───┐
       ▼              ▼              ▼           ▼
   Branch 0       Branch 1      Branch 2    Branch 255
       │              │              │           │
  Linear(→20)    Linear(→20)   Linear(→20)  Linear(→20)
     ReLU           ReLU          ReLU         ReLU
  Linear(→10)    Linear(→10)   Linear(→10)  Linear(→10)
     ReLU           ReLU          ReLU         ReLU
  Linear(→2)     Linear(→2)    Linear(→2)   Linear(→2)
       │              │              │           │
       ▼              ▼              ▼           ▼
   Logits k=0    Logits k=1    Logits k=2   Logits k=255
```

## Acknowledgments

### Dataset
- **ASCAD Dataset**: [ANSSI-FR/ASCAD](https://github.com/ANSSI-FR/ASCAD)
  - Prouff, E., Strullu, R., Benadjila, R., Cagli, E., & Dumas, C. (2018). Study of deep learning techniques for side-channel analysis and introduction to ASCAD database. *IACR Cryptology ePrint Archive*.

### Research Papers
- **Multi-Output Classification Attack**: Kim, J., et al. (2019). Efficient Non-profiled Side-Channel Attack Using Multi-Output Classification Neural Network. *IACR Transactions on Cryptographic Hardware and Embedded Systems*.

- **Differential Power Analysis**: Kocher, P., Jaffe, J., & Jun, B. (1999). Differential power analysis. *CRYPTO 1999*.

### Frameworks
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [h5py](https://www.h5py.org/) - HDF5 file handling

## License

This project is for educational and research purposes.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{multi_output_sca,
  author = {Your Name},
  title = {Multi-Output Neural Network for Side-Channel Analysis},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/multi_output_SCA}
}
```
