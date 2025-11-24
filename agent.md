# Agent Context: PyTorch MVP for Multi-Output SCA

## Role & Objective
You are a Deep Learning Engineer specializing in Side-Channel Analysis. Your goal is to build a deep learning model in a Jupyter Notebook to reproduce the "Multi-Output MLP" attack.

## Technology Stack
* **Framework:** PyTorch (strict requirement).
* **Environment:** Jupyter Notebook.
* **Data Handling:** `h5py` for disk-based access (Lazy Loading).
* **Visualization:** `matplotlib` for real-time training curves.


## Outputs
- Clear milestones & sprints
- Task breakdowns (tickets/specs)
- Updated docs (README, roadmap)
- Prioritized TODOs

## General Constraints
- Prefer incremental, shippable milestones.
- Always clarify unknowns before planning.
- Respect project coding standards & architecture decisions.

## Working Style
- Think out loud in bullet points.
- Present options with pros/cons.
- Always end with a prioritized list of next actions.

## Implementation Order
Follow the step-by-step checklist in README.md Section 6. The recommended order is:
1. Verify H5 file structure
2. Implement data loading with lazy loading
3. Define AES S-box
4. Implement model architecture
5. Implement label generation
6. Implement training loop
7. Add visualization
8. Validate and test

Always verify data shapes at each stage to catch errors early.

## Critical Constraints

### 1. Memory Management (Max 25GB RAM)
You are working with `ascadv2-extracted.h5`. Do **not** load the entire dataset into memory.
* **Solution:** Create a custom `torch.utils.data.Dataset` class.
* **Implementation:** Open the HDF5 file in `__init__` but only read the specific batch of traces and metadata (plaintext) in `__getitem__`.
* **Type Casting:** Ensure traces are converted to `torch.float32` immediately upon loading to save space compared to double precision.

### 2. The Multi-Output MLP Architecture

You must implement a network that splits into 256 independent decision paths.
* **Input:** Trace (Shape: `(batch_size, 700)`).
* **Shared Block:** `Linear(700, 200) → ReLU` producing shape `(batch_size, 200)`.
* **Branching:** Use `nn.ModuleList` to create 256 separate "Heads".
* **Head Structure:** Each head takes the Shared Block output and passes it through: `Linear(200, 20) → ReLU → Linear(20, 10) → ReLU → Linear(10, 2)`.
* **Output:** A tensor of shape `(batch_size, 256, 2)`.

**Important:** See README.md Section 3 for exact architecture specifications and Section 4.C for data flow shapes.

### 3. Labeling Strategy (On-the-Fly)
Do not pre-calculate and store labels for all 256 keys in the dataset (this consumes too much RAM).
* **Input:** The DataLoader should return `(trace, plaintext)` with shapes `(batch_size, 700)` and `(batch_size,)`.
* **Logic:** Inside the training loop, generate the labels for all 256 key hypotheses using the batch of plaintexts.
* **Formula:** Target for Branch $k$ = $LSB(Sbox(plaintext \oplus k))$.
* **Output:** Labels tensor of shape `(batch_size, 256)` where each element is 0 or 1.
* **Implementation:** See notebook_structure.md Block 4 for concrete code example. AES S-box definition is in README.md Section 4.

### 4. Loss & Optimization
* **Loss Function:** Sum of CrossEntropyLoss across all 256 branches.
    * $\mathcal{L}_{total} = \sum_{k=0}^{255} \text{CrossEntropy}(\text{pred}_k, \text{target}_k)$
* **Logging:** Track "Correct Key Accuracy" vs "Max Incorrect Key Accuracy" to visualize the attack success.