### Multi Output Model for Side Channel Analysis
- This notebook is an attempt at a replication of Van-Phuc Hoang et al.s work "Efficient Nonprofiled Side-Channel Attack Using
Multi-Output Classification Neural Network"


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
- *Cursor AI IDE and Opus 4.5 was used to help with many facets of this project* 

### References
[1] V.-P. Hoang, N.-T. Do, and V. S. Doan, “Efficient Nonprofiled Side-Channel Attack Using Multi-Output Classification Neural Network,” IEEE Embedded Syst. Lett., vol. 15, no. 3, pp. 145–148, Sept. 2023, doi: 10.1109/LES.2022.3213443.
[2] M.-L. Akkar and C. Giraud, “An Implementation of DES and AES, Secure against Some Attacks,” in Cryptographic Hardware and Embedded Systems — CHES 2001, vol. 2162, Ç. K. Koç, D. Naccache, and C. Paar, Eds., in Lecture Notes in Computer Science, vol. 2162. , Berlin, Heidelberg: Springer Berlin Heidelberg, 2001, pp. 309–318. doi: 10.1007/3-540-44709-1_26.
[3] B. Timon, “Non-Profiled Deep Learning-based Side-Channel attacks with Sensitivity Analysis,” TCHES, pp. 107–131, Feb. 2019, doi: 10.46586/tches.v2019.i2.107-131.


