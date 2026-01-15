
import os
import numpy as np
import pandas as pd
import bearing_utils

# =============================================================================
# CONFIGURATION
# =============================================================================
# Base path for input Normal files
INPUT_PATH = r'CWRU_Bearing_NumPy-main'
RPMS = [1730, 1750, 1772, 1797]

# Fault Parameters
FAULT_TYPES = ['Inner', 'Outer', 'Ball']
K_VALUES = [1.0, 5.0, 10.0, 50.0]  # Severity factor
PHASES = [0, np.pi/2, np.pi, 3*np.pi/2] # Phase shifts

# Signal Parameters
FS = 12000  # Sampling frequency (Hz)
SEGMENT_LEN = 4096 # Points per segment
FAULT_DIAMETERS = [0.178, 0.356, 0.533, 0.711, 1.016] # Fault diameters in mm

# Output
OUTPUT_DIR = 'synthetic_dataset_output'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'df_synthetic.pkl')

def get_normal_data_path(rpm):
    """Constructs file path for Normal data given RPM."""
    # Assuming standard folder structure based on ls output
    # CWRU_Bearing_NumPy-main/1730 RPM/1730_Normal.npz
    return os.path.join(INPUT_PATH, f"{rpm} RPM", f"{rpm}_Normal.npz")

def load_normal_segments(rpm, max_segments=None):
    """Loads Normal data for specific RPM and segments it."""
    file_path = get_normal_data_path(rpm)
    try:
        data = np.load(file_path)
        # Assuming 'DE' key contains the Drive End data
        if 'DE' in data:
            signal = data['DE'].ravel()
        else:
            # Fallback or error if key structure differs
            keys = list(data.keys())
            print(f"Warning: 'DE' key not found in {file_path}. Available keys: {keys}")
            return []
            
        # Segment the signal
        segments = []
        n_samples = len(signal)
        step = SEGMENT_LEN # No overlap for dataset diversity, or change as needed
        
        for i in range(0, n_samples - SEGMENT_LEN, step):
            seg = signal[i : i + SEGMENT_LEN]
            segments.append(seg)
            if max_segments and len(segments) >= max_segments:
                break
        return segments
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def synthesize_fault_signal(fault_type, rpm, fault_diameter, K, phase, duration, t):
    """
    Generates the pure synthetic fault signal (time domain) using Tandon's model.
    """
    signal = np.zeros_like(t)
    
    # 1. Get Spectrum from bearing_utils
    # Note: bearing_utils functions return a DataFrame with 'Frequency_Hz' and 'Amplitude...'
    if fault_type == 'Inner':
        # Using a fixed reasonable number of harmonics/sidebands for synthesis
        df_spec = bearing_utils.calcular_espectro_inner_completo(
            fault_diameter, rpm, max_harmonics=5, num_sidebands=3, K=K
        )
        # Column name check (bearing_utils output format)
        amp_col = 'Amplitude_Accel_m_s2'
        
    elif fault_type == 'Outer':
        df_spec = bearing_utils.calcular_espectro_outer_race(
            fault_diameter, rpm, max_harmonics=10, K=K
        )
        amp_col = 'Amplitude_m_s2'
        
    elif fault_type == 'Ball':
        df_spec = bearing_utils.calcular_espectro_ball_completo(
            fault_diameter, rpm, max_harmonics=5, num_sidebands=3, K=K
        )
        amp_col = 'Amplitude_Accel_m_s2'
    else:
        return signal

    # 2. Reconstruct Time Signal (Sum of Cosines with Phase)
    # y(t) = Sum( Amp * cos(2*pi*f*t + phase) )
    for _, row in df_spec.iterrows():
        freq = row['Frequency_Hz']
        amp = row[amp_col]
        # Adding phase shift to the argument
        signal += amp * np.cos(2 * np.pi * freq * t + phase)
        
    return signal

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    dataset_rows = []
    
    # Pre-calculate time vector for one segment
    t = np.arange(0, SEGMENT_LEN / FS, 1/FS)
    duration = SEGMENT_LEN / FS

    print(f"Starting Synthetic Dataset Generation...")
    print(f"RPMs: {RPMS}")
    print(f"Faults: {FAULT_TYPES}")
    print(f"Ks: {K_VALUES}")
    
    total_generated = 0

    for rpm in RPMS:
        print(f"\nProcessing RPM: {rpm}")
        
        # 1. Load Normal Backgrounds
        # Limit segments per RPM to keep dataset balanced/manageable? 
        # For now, let's take up to 50 segments to serve as diversity base.
        normal_segments = load_normal_segments(rpm, max_segments=50)
        
        if not normal_segments:
            continue
            
        print(f"  Loaded {len(normal_segments)} normal segments.")

        # 2. Iterate through variations
        for normal_seg in normal_segments:
            
            # --- A. Pure Normal (K=0) ---
            # Ideally we add this once per segment
            dataset_rows.append({
                'signal': normal_seg,
                'rpm': rpm,
                'fault_type': 'Normal',
                'fault_diameter': 0.0,
                'K': 0.0,
                'phase': 0.0,
                'is_synthetic': False,
                'base_signal_idx': str(total_generated) # Just a unique tracking ID concept
            })
            total_generated += 1
            
            # --- B. Synthetic Faults ---
            for f_type in FAULT_TYPES:
                for f_diam in FAULT_DIAMETERS:
                    for K in K_VALUES:
                        for phi in PHASES:
                            
                            # Generate pure fault signal
                            fault_sig = synthesize_fault_signal(f_type, rpm, f_diam, K, phi, duration, t)
                        
                        # Add to normal background
                        # Ensure lengths match (should be 4096)
                        if len(fault_sig) != len(normal_seg):
                            # Truncate to match if needed
                            min_len = min(len(fault_sig), len(normal_seg))
                            hybrid_sig = normal_seg[:min_len] + fault_sig[:min_len]
                        else:
                            hybrid_sig = normal_seg + fault_sig
                            
                        dataset_rows.append({
                            'signal': hybrid_sig,
                            'rpm': rpm,
                            'fault_type': f_type,
                            'fault_diameter': f_diam,
                            'K': K,
                            'phase': phi,
                            'is_synthetic': True,
                            'base_signal_idx': str(total_generated)
                        })
                        total_generated += 1
        
        print(f"  Current total rows: {len(dataset_rows)}")

    # 3. Save
    print(f"\nSaving dataframe with {len(dataset_rows)} samples...")
    df = pd.DataFrame(dataset_rows)
    df.to_pickle(OUTPUT_FILE)
    print(f"Saved to: {OUTPUT_FILE}")
    
    # 4. Save CSV preview (without signals) for quick detail check
    df_meta = df.drop(columns=['signal'])
    preview_csv = os.path.join(OUTPUT_DIR, 'synthetic_metadata.csv')
    df_meta.to_csv(preview_csv, index=False)
    print(f"Metadata saved to: {preview_csv}")

if __name__ == "__main__":
    main()
