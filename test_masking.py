"""
Test script to analyze ASCAD Boolean masking and determine correct leakage model.

ASCAD uses Boolean masking where:
- The masked intermediate leaks: Sbox(p XOR k) XOR mask

This script tests different leakage models to see which recovers the key.
"""

import h5py
import numpy as np

# AES S-box
SBOX = np.array([
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

TARGET_BYTE = 2
N_TRACES = 20000

def hamming_weight(x):
    """Compute Hamming weight of byte array."""
    return np.unpackbits(x.reshape(-1, 1).astype(np.uint8), axis=1).sum(axis=1)

def get_max_corr(traces, labels):
    """Get max absolute correlation across all sample points (vectorized)."""
    # Vectorized correlation computation - much faster
    traces_centered = traces - traces.mean(axis=0)
    labels_centered = labels - labels.mean()
    
    # Correlation for each sample point
    numerator = (traces_centered * labels_centered[:, None]).mean(axis=0)
    denominator = traces_centered.std(axis=0) * labels_centered.std()
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        corrs = np.abs(numerator / denominator)
        corrs = np.nan_to_num(corrs, nan=0.0)
    
    return corrs.max()

def main():
    print("Loading ASCAD dataset...")
    with h5py.File("ASCAD.h5", "r") as f:
        traces = f["Profiling_traces/traces"][:N_TRACES].astype(np.float32)
        metadata = f["Profiling_traces/metadata"][:N_TRACES]
    
    plaintexts = metadata["plaintext"][:, TARGET_BYTE]
    masks = metadata["masks"][:, TARGET_BYTE]
    true_key = int(metadata["key"][0, TARGET_BYTE])
    
    print(f"True key: {true_key} (0x{true_key:02x})")
    print(f"Traces shape: {traces.shape}")
    print(f"Mask byte index: {TARGET_BYTE}")
    
    # Check masks structure
    print(f"\n=== Mask Analysis ===")
    print(f"Masks shape per trace: {metadata['masks'].shape}")
    print(f"First trace masks: {metadata['masks'][0]}")
    print(f"Unique mask values at byte {TARGET_BYTE}: {len(np.unique(masks))}")
    
    # Normalize traces
    traces = traces - np.mean(traces)
    traces = traces / np.max(np.abs(traces))
    
    print("\nComputing correlations for all 256 key hypotheses...")
    print("(This may take a minute...)\n")
    
    # Test different leakage models
    models = {}
    
    # Model 1: Unmasked LSB
    print("Testing: UNMASKED LSB = LSB(Sbox(p XOR k))")
    unmasked_lsb_corrs = np.zeros(256)
    for k in range(256):
        labels = SBOX[plaintexts ^ k] & 1
        unmasked_lsb_corrs[k] = get_max_corr(traces, labels)
    models['Unmasked LSB'] = unmasked_lsb_corrs
    
    # Model 2: Masked LSB  
    print("Testing: MASKED LSB = LSB(Sbox(p XOR k) XOR mask)")
    masked_lsb_corrs = np.zeros(256)
    for k in range(256):
        labels = (SBOX[plaintexts ^ k] ^ masks) & 1
        masked_lsb_corrs[k] = get_max_corr(traces, labels)
    models['Masked LSB'] = masked_lsb_corrs
    
    # Model 3: Unmasked Hamming Weight
    print("Testing: UNMASKED HW = HW(Sbox(p XOR k))")
    unmasked_hw_corrs = np.zeros(256)
    for k in range(256):
        labels = hamming_weight(SBOX[plaintexts ^ k])
        unmasked_hw_corrs[k] = get_max_corr(traces, labels)
    models['Unmasked HW'] = unmasked_hw_corrs
    
    # Model 4: Masked Hamming Weight
    print("Testing: MASKED HW = HW(Sbox(p XOR k) XOR mask)")
    masked_hw_corrs = np.zeros(256)
    for k in range(256):
        sbox_out = SBOX[plaintexts ^ k]
        masked_out = (sbox_out ^ masks).astype(np.uint8)
        labels = hamming_weight(masked_out)
        masked_hw_corrs[k] = get_max_corr(traces, labels)
    models['Masked HW'] = masked_hw_corrs
    
    # Model 5: Full Sbox output (all 8 bits via HW)
    print("Testing: Full Sbox output correlation")
    full_sbox_corrs = np.zeros(256)
    for k in range(256):
        labels = SBOX[plaintexts ^ k]
        full_sbox_corrs[k] = get_max_corr(traces, labels)
    models['Full Sbox'] = full_sbox_corrs
    
    # Model 6: Masked full Sbox output
    print("Testing: Masked Full Sbox output correlation")
    masked_full_corrs = np.zeros(256)
    for k in range(256):
        labels = SBOX[plaintexts ^ k] ^ masks
        masked_full_corrs[k] = get_max_corr(traces, labels)
    models['Masked Full'] = masked_full_corrs
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS: Key Ranking by Leakage Model")
    print("="*70)
    print(f"{'Model':<20} {'True Key Corr':<15} {'Best Key':<12} {'Best Corr':<12} {'Rank':<8}")
    print("-"*70)
    
    for name, corrs in models.items():
        rank = int((corrs > corrs[true_key]).sum() + 1)
        best_key = int(np.argmax(corrs))
        print(f"{name:<20} {corrs[true_key]:<15.4f} {best_key:<12} {corrs[best_key]:<12.4f} {rank:<8}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Find best model
    best_model = None
    best_rank = 257
    for name, corrs in models.items():
        rank = int((corrs > corrs[true_key]).sum() + 1)
        if rank < best_rank:
            best_rank = rank
            best_model = name
    
    if best_rank == 1:
        print(f"✓ '{best_model}' successfully recovers the key!")
    else:
        print(f"Best performing model: '{best_model}' (rank {best_rank})")
        print("""
For masked implementations, simple CPA often fails because:
1. The mask adds noise that averages out the correlation
2. Second-order attacks may be needed (combining multiple leakage points)

However, neural network approaches can still work because they:
- Learn complex nonlinear relationships
- Can implicitly combine first and second-order leakage
- Don't require explicit mask knowledge for training (non-profiled)

The CNNMO paper's approach should work even with these low correlations.
""")
    
    # Additional analysis: Check if masks are correctly aligned
    print("\n=== Sanity Check: ASCAD Label Verification ===")
    with h5py.File("ASCAD.h5", "r") as f:
        ascad_labels = f["Profiling_traces/labels"][:10]
    
    print("ASCAD provides labels = Sbox(p XOR k) for the TRUE key")
    print(f"First 10 ASCAD labels: {ascad_labels}")
    computed = SBOX[plaintexts[:10] ^ true_key]
    print(f"Our computed Sbox out: {computed}")
    print(f"Match: {np.array_equal(ascad_labels, computed)}")

if __name__ == "__main__":
    main()

