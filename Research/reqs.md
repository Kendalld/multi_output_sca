# Requirements Document

## 1. Datasets and labeling

### 1.1 Underlying datasets

Both projects use:

- **ASCAD fixed-key dataset** (software AES on 8-bit AVR, first-order masked)
  - 700 samples per trace
  - Target leakage: output of 3rd S-box byte with an unknown mask

> **Note:** You'll want to align your trace windows with theirs (same encryption phase) for best comparability.

### 1.2 Labeling strategy

Timon uses binary labels derived from the S-box output; this is reused by Hoang et al.:

Let `p_i` be plaintext byte for trace `i`, `k_j` a key hypothesis, and `Sbox` the AES S-box.

Define intermediate value:

```
v_{i,j} = Sbox(p_i ⊕ k_j)
```

Then the label for hypothesis `j` and trace `i` in the multi-output paper is:

```
l_{i,j} = LSB(v_{i,j})  (either 0 or 1)
```

### 1.3 Multi-output dataset structure (Hoang et al.)

From a standard single-output SCA dataset with N traces:

- For each trace `i` (1…N) and each key hypothesis `k_j` (0…255), compute `l_{i,j}` as above
- Build a label matrix of size N × 256, where row `i` is a 256-dim vector of 0/1 labels

They define several reconstructed datasets from ASCAD – approximate structure:

- **Dataset1:** ASCAD single-output (for baseline MLPDDLA, ~20k traces × 700 samples)
- **Dataset2:** ASCAD multi-output (same traces, 256-length label vectors)
- **Dataset3:** ASCAD multi-output, more traces (50k × 700) for harder noise conditions

> **Note:** Exact counts in the table: ASCAD D1/D2: 20,000 traces, 700 samples; D3: 50,000. Labels are LSB vs LSB-vector depending on single vs multi-output.

### 1.4 Simulated countermeasures

#### 1.4.1 Noise generation (ASCAD)

They add i.i.d. Gaussian noise sample-wise:

```
t_noise(i,m) = t(i,m) + σ ⋅ randn(1,m) + mean
```

where `randn` is standard normal; `mean = 0`.

They use `σ ∈ {0.5, 1.0, 1.5}`.

Rebuild datasets `DatasetX-N1`, `DatasetX-N2`, `DatasetX-N3` (X ∈ {1,2,3}) with these added.


### 1.5 Input normalization

From Timon (and you should keep this in your reimplementation):

1. Remove the mean of all traces
2. Scale each sample to [-1, 1]

> **Note:** Hoang et al. do not restate normalization, but they are benchmarking against DDLA, so using the same normalization is safest.

---

## 2. Original DDLA implementation (Timon 2019)

These are the single-output networks and training settings that the 2023 letter compares against.

### 2.1 Global training hyperparameters

Used for all experiments:

- **Framework:** PyTorch 0.4.1
- **Loss:** Mean Squared Error (MSE) between one-hot ground truth and network output
- **Accuracy metric:** proportion of correctly classified samples
- **Batch size:** 1000
- **Learning rate:** 0.001 (constant, no decay)
- **Optimizer:** Adam with β₁ = 0.9, β₂ = 0.999, ε = 1e-8, no LR decay
- **Initialization:** default PyTorch (typically Kaiming/He for ReLU)

**Epoch counts:**

- Simulated traces: 50 epochs
- ASCAD experiments: commonly 50–100 epochs (e.g., 50 for ASCAD)

> **Note:** For exact replication, mirror his epoch counts for each experiment type, but Hoang et al. often re-report timings for 10 and 30 epochs to compare efficiency.

### 2.2 Network architectures

All DDLA models are binary classifiers for a single key guess (2-class: bit is 0 or 1).

#### 2.2.1 MLPsim (for simple simulations)

- **Input:** n samples per trace (e.g., 50 in simulations)
- **Hidden layer 1:** Dense(70, activation=ReLU)
- **Hidden layer 2:** Dense(50, activation=ReLU)
- **Output:** Dense(2, activation=Softmax)

#### 2.2.2 CNNsim (for simulated misalignment)

- **Conv1:** 1D convolution, 8 filters, kernel size 8, stride 1, no padding, ReLU
- **MaxPool1:** pool size 2
- **Conv2:** 1D convolution, 4 filters, kernel size 4, stride 1, no padding, ReLU
- **MaxPool2:** pool size 2
- **Dense output:** 2 units, Softmax

#### 2.2.3 MLPexp (used on ASCAD)

This is the core "MLPDDLA" architecture that Hoang et al. use as the per-branch network:

- **Input:** full trace window (e.g., 500 or 700 samples)
- **Hidden 1:** Dense(20, ReLU)
- **Hidden 2:** Dense(10, ReLU)
- **Output:** Dense(2, Softmax)

#### 2.2.4 CNNexp (used on ASCAD)

Also called "CNNDDLA" in the later paper:

- **Conv1:** 1D conv with 4 filters, kernel size 32, stride 1, no padding, ReLU
- **AvgPool1:** pool size 2
- **BatchNorm1**
- **Conv2:** 1D conv with 4 filters, kernel size 16, stride 1, no padding, ReLU
- **AvgPool2:** pool size 4
- **BatchNorm2**
- **Flatten**
- **Dense output:** 2 units, Softmax

### 2.3 DDLA attack loop

Timon's DDLA (Algorithm 1):

```
For each key hypothesis k:
  1. Re-initialize Net's weights
  2. Compute labels H_{i,k} = h(F(p_i, k)), e.g. MSB or LSB
  3. Train Net on all traces for ne epochs with these labels
  4. Record metrics (loss, accuracy, sensitivity) for this key

At the end, select the key whose run:
  - Has lowest loss / highest accuracy, and
  - (Optionally) whose sensitivity maps show clear leakage
```

> **Note:** Hoang et al. keep exactly this per-key model as the branch architecture inside their multi-output networks.

---

## 3. Multi-output models (Hoang et al. 2023)

They introduce two architectures:

- **MLPMO** – multi-output MLP for masking + noise
- **CNNMO** – multi-output CNN for ASCAD traces

Each has 256 branches, one per key hypothesis, sharing some layers.

Training is done in Keras (TensorFlow backend) on a CPU (Intel i5-9500, 24 GB RAM).

### 3.1 General training settings

From the letter:

- **Framework:** Keras
- **Optimizer:** Adam with default settings (so lr = 0.001, β₁=0.9, β₂=0.999, ε≈1e-7 or framework default)

**Loss per branch:** standard 2-class cross-entropy:

```
L[k](θ) = -1/N_s ∑_{j=1}^2 y_true log(z)
```

where `z` is branch output probability, `N_s` number of training traces.

**Total loss:** sum of branch losses since all γₖ = 1:

```
L_total = ∑_{k=1}^{256} γ_k L[k](θ),  γ_k = 1
```

- **Accuracy metrics:** they keep per-branch accuracy curves; correct key shows clear separation
- **Weight initialization:** He-uniform (for both MLPMO and CNNMO)

**Batch sizes from Table II:**

- MLPMO batch size: 1000
- CNNMO batch size: 50

**Typical epoch counts in experiments:**

- Masking/noise: 10, 25, and 30 epochs (they plot up to 25; time comparisons for 10 and 30)
- CNNMO: around 50-100 epochs when comparing against CNNDDLA, but CNNMO shows key separation very early (fewer epochs needed)

> **Note:** For reproduction, pick 30 epochs for masking/noise experiments and 50-100 epochs for CNNMO, then you can sub-sample to 10/20/30 when computing the same timing plots.

### 3.2 MLPMO architecture (multi-output MLP)

#### 3.2.1 Top-level structure

- **Input size:** length of trace (ASCAD → 700 samples)

**Shared layer (optional):**

- Dense layer with ReLU and 0, 50, 200, or 400 neurons

**Model names:**

- **Non-SoSL:** no shared layer (0 neurons)
- **SoSL-50 / SoSL-200 / SoSL-400:** shared layers of corresponding size

**Branches:** 256 parallel branches (one per key hypothesis). All share the same architecture and have distinct weights.

#### 3.2.2 Branch architecture (per key hypothesis)

Each branch is literally MLPDDLA without its input layer:

Given an input vector (either raw trace or shared-layer output):

- Dense(20, activation=ReLU)
- Dense(10, activation=ReLU)
- Dense(2, activation=Softmax)

He-uniform initialization is applied to these dense layers.

#### 3.2.3 Notes

- When no shared layer, the branch's first Dense(20) is fully connected to the original input
- When using a shared layer, that shared layer feeds all branches
- All branch weights are updated on every mini-batch, unlike classic DDLA which only trains one key at a time

### 3.3 CNNMO architecture (multi-output CNN)

Used for ASCAD traces with CNN-based feature extraction.

#### 3.3.1 Shared CNN

- **Input size:** 700 samples per trace (ASCAD)

**Two shared "blocks", each:**

- **Conv1D**
  - First block: 4 filters, kernel size 32, stride 1, no padding
  - Second block: 4 filters, kernel size 16, stride 1, no padding
- **BatchNorm**
- **AveragePooling1D**
  - Pool size 2 (first block)
  - Pool size 4 (second block)
- **ReLU activation**

Ordered as: `conv1d → norm → pool → relu`

After these two blocks: **Flatten**

> **Note:** This shared part is CNNexp/CNNDDLA minus the final dense layer, just reused for all keys.

#### 3.3.2 Branches

256 branches. For each branch `k`, after the shared Flatten:

- Dense(2, activation=Softmax), He-uniform initializer

No additional hidden layers inside the branch are mentioned; so each branch is just a small head over the shared CNN features.

---

## 4. Attack logic & evaluation

### 4.1 How the multi-output networks replace DDLA loops

In DDLA (Timon), you train a separate 2-class network per key; each training gives you a curve of loss/accuracy/sensitivity, and you pick the best key.

Hoang et al. emulate this in one shot:

- **Input:** original traces
- **Output:** for each trace, the model produces 256 two-class outputs (one for each key)
- **Per-branch loss & accuracy** are computed directly from the Keras multi-output graph because they use one cross-entropy per branch (multi-loss)

**So you can:**

1. Record per-branch accuracy (or loss) vs epoch
2. Select the key whose branch has:
   - Highest final accuracy or
   - Lowest final loss,
   - with a stable gap from others

**They show:**

- For masking: clear accuracy separation from epoch ~5 onward (SoSL-200) and even earlier for Non-SoSL
- For CNNMO: loss curve for correct key separates quickly; CNNMO shows improved performance compared to CNNDDLA with faster training

### 4.2 Success-rate evaluation (noise experiments)

For combined masking + noise:

1. Build noisy datasets for `σ ∈ {0.5, 1.0, 1.5}`
2. For each `σ` and each model (MLPDDLA, Non-SoSL, SoSL-200):
   - Repeat the attack 50 times with different random seeds/noise
   - Compute success rate = #successful attacks / 50

**They show:**

- At σ=0.5: all reach 100%
- At σ=1.0 and 1.5: SoSL-200 keeps ≥20% higher success rate than MLPDDLA (e.g., 96% vs 80% at σ=1.0; 44–36% vs 30% at σ=1.5 for 20k traces)
- With 50k traces and σ=1.5 (Dataset3-N3), SoSL-200 recovers 100% success

### 4.3 Timing comparisons

Timing is on a single CPU machine (i5-9500, 24 GB). All models in their work are run under similar conditions.

**Key timing points:**

**On masked ASCAD data (20k traces), 10 epochs:**

- MLPDDLA (classic DDLA loop): ~596 s
- Non-SoSL (MLPMO no shared layer): ~72 s
- SoSL-200: ~65 s
- → About 8–9× speedup

**30 epochs:**

- Still around 6–7× speedup vs DDLA

**CNNMO (ASCAD, ~20k traces, 50-100 epochs):**

- CNNDDLA: significantly slower due to per-key training loop
- CNNMO: faster training with shared CNN layers and parallel branch updates
- → Significant speedup over DDLA approach

**To replicate these numbers closely, you'll want:**

- Same dataset sizes (e.g., exactly 20k traces for ASCAD)
- Same batch sizes (1000 for MLP, 50 for CNN)
- Same number of epochs (they report 10, 30 for MLP; 50-100 for CNN)

---

## 5. Practical re-implementation checklist

Here's a compact "to-do" list you can treat as a spec:

### Preprocessing

- Load ASCAD fixed-key traces with trace length 700 samples
- Center and scale traces to [-1, 1] (global mean + scaling)

### Label generation

- For each plaintext byte and key hypothesis 0…255, compute `LSB(Sbox(p ⊕ k))`
- Build label matrix N × 256 with entries in {0,1}

### Noise

- Add Gaussian noise with `σ∈{0.5,1.0,1.5}` per sample

### Networks

- Implement MLPexp & CNNexp as in §2.2 for single-output DDLA baselines
- Implement MLPMO & CNNMO as in §3 with 256 branches, He-uniform init, Adam optimizer

### Training

- **For DDLA:** loop over all 256 keys, re-init net each time, train for ne epochs with batch=1000 (MLP) or similar; log metrics
- **For MO:** single training per dataset:
  - MLPMO: batch=1000, epochs={10,30}
  - CNNMO: batch=50, epochs≈50-100

### Metrics & evaluation

- Record per-branch accuracy (masking/noise) and/or loss (de-sync) vs epoch
- Select key with best metric as recovered key
- For success rates, repeat 50 runs for noisy experiments
