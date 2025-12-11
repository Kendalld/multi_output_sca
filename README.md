### Multi Output Model for Side Channel Analysis
- Notebook-first project for side-channel analysis (SCA) experiments using a multi-output neural network.


### Datasets
- Local ASCAD file used by the notebook: `datasets/ASCADv1/ASCAD.h5`
- ASCAD reference: [ASCAD v1: fixed key](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key)

### Setup
- Use Python 3.10+.
- Create/activate a virtual environment.
- Install dependencies from `requirements.txt`.
- Run Jupyter from the project root (requirements include `notebook`, `ipykernel`, `ipywidgets`).


### Data pipeline
- Traces: `Profiling_traces/traces` from ASCAD, sliced with `NUM_TRACES` and `TRACE_START_IDX`.
- Metadata: plaintext and key read from `Profiling_traces/metadata`.
- Target byte: `TARGET_BYTE_IDX = 2`. 3rd byte p ^ k paring confirmed leakage
- Trace normalization: per-sample standardization \( (x - \mu) / \sigma \).
- Model input tensor shape: `(N, 700)` float32.

### Labels and leakage model
- 256 label sets are generated, one per key candidate \(k \in [0,255]\).
- For each trace: `val = AES_Sbox[plaintext ^ k]` and the label is `val & 1` for LBS
- Label tensor shape: `(N, 256)` with dtype long.
- Classification task: binary (LSB leakage model), `OUTPUT_CLASSES = 2`.

### Model 
- Multilayer Perceptron
- Shared trunk: optional `Linear(input_dim -> SHARED_DIM) + ReLU`, or `Identity` if `SHARED_DIM = 0`.
- Branches: 256 independent MLP heads, each `Linear(branch_input -> 20) + ReLU + Linear(20 -> 10) + ReLU + Linear(10 -> 2)`.
- Output: logits stacked to shape `(batch, 256, 2)`.
- Default hyperparameters in the notebook: 
    - `SHARED_DIM=0`
    - `BRANCH_LAYER_1=20`
    - `BRANCH_LAYER_2=10`
    - `LEARNING_RATE=5e-4`
    - `BATCH_SIZE=100`
    - `NUM_EPOCHS=30`

### Training and key ranking
- Loss per branch: cross-entropy on `output[:, k, :]` vs `target[:, k]`.
- Total loss: mean over all 256 branches (divide by 256 for stability).
- Optimizer: Adam with learning rate `LEARNING_RATE`.
- Key scoring: compute average loss per branch and convert to a score via `score = -loss`.
- Ranking: sort the 256 scores descending and track the rank of the real key byte.

### Notes
- `SHARED_DIM > 0` can reduce total compute by sharing features across all 256 branches. `200` worked well
- To switch leakage bit (example): replace `val & 1` with an MSB-style label such as `(val >> 7) & 1`.
- A CNN option is viable and would work much better for traces that aren't aligned

