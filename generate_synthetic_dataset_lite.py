
import os
import numpy as np
import pandas as pd
import bearing_utils

# =============================================================================
# CONFIGURATION - LITE VERSION
# =============================================================================
INPUT_PATH = r'CWRU_Bearing_NumPy-main'
RPMS = [1730, 1750, 1772, 1797]
FAULT_TYPES = ['Inner', 'Outer', 'Ball']
K_VALUES = [1.0, 5.0, 10.0, 50.0]
PHASES = [0, np.pi/2, np.pi, 3*np.pi/2]
FS = 12000
SEGMENT_LEN = 4096
FAULT_DIAMETERS = [0.178, 0.356, 0.533, 0.711, 1.016] 
OUTPUT_DIR = 'synthetic_dataset_output'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'df_synthetic_LITE.pkl')

def get_normal_data_path(rpm):
    return os.path.join(INPUT_PATH, f"{rpm} RPM", f"{rpm}_Normal.npz")

def load_normal_segments(rpm, max_segments=None):
    file_path = get_normal_data_path(rpm)
    try:
        data = np.load(file_path)
        if 'DE' in data:
            signal = data['DE'].ravel() # Ensure Flat
        else:
            return []
        segments = []
        n_samples = len(signal)
        step = SEGMENT_LEN 
        for i in range(0, n_samples - SEGMENT_LEN, step):
            seg = signal[i : i + SEGMENT_LEN]
            segments.append(seg)
            if max_segments and len(segments) >= max_segments:
                break
        return segments
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def synthesize_fault_signal(fault_type, rpm, fault_diameter, K, phase, duration, t):
    signal = np.zeros_like(t)
    if fault_type == 'Inner':
        df_spec = bearing_utils.calcular_espectro_inner_completo(fault_diameter, rpm, 5, 3, 20, K=K)
        amp_col = 'Amplitude_Accel_m_s2'
    elif fault_type == 'Outer':
        df_spec = bearing_utils.calcular_espectro_outer_race(fault_diameter, rpm, 10, K=K)
        amp_col = 'Amplitude_m_s2'
    elif fault_type == 'Ball':
        df_spec = bearing_utils.calcular_espectro_ball_completo(fault_diameter, rpm, 5, 3, 20, K=K)
        amp_col = 'Amplitude_Accel_m_s2'
    else:
        return signal

    for _, row in df_spec.iterrows():
        signal += row[amp_col] * np.cos(2 * np.pi * row['Frequency_Hz'] * t + phase)
    return signal

def main():
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
        except:
            pass # Exists
        
    dataset_rows = []
    t = np.arange(0, SEGMENT_LEN / FS, 1/FS)
    duration = SEGMENT_LEN / FS

    print(f"Starting LITE Synthetic Generation (1 segment per RPM)...")
    
    total_generated = 0
    for rpm in RPMS:
        print(f"Processing RPM: {rpm}")
        # LIMIT TO 1 SEGMENT
        normal_segments = load_normal_segments(rpm, max_segments=1)
        if not normal_segments:
            continue
            
        for normal_seg in normal_segments:
            dataset_rows.append({
                'signal': normal_seg,
                'rpm': rpm,
                'fault_type': 'Normal',
                'fault_diameter': 0.0,
                'K': 0.0,
                'phase': 0.0,
                'is_synthetic': False
            })
            for f_type in FAULT_TYPES:
                for f_diam in FAULT_DIAMETERS:
                    for K in K_VALUES:
                        for phi in PHASES:
                            fault_sig = synthesize_fault_signal(f_type, rpm, f_diam, K, phi, duration, t)
                            if len(fault_sig) != len(normal_seg):
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
                                'is_synthetic': True
                            })
        print(f"  Rows so far: {len(dataset_rows)}")

    print(f"\nSaving LITE dataframe with {len(dataset_rows)} samples...")
    df = pd.DataFrame(dataset_rows)
    df.to_pickle(OUTPUT_FILE)
    print(f"Saved to: {OUTPUT_FILE}")
    
    # Save CSV preview
    df_meta = df.drop(columns=['signal'])
    preview_csv = os.path.join(OUTPUT_DIR, 'synthetic_metadata_LITE.csv')
    df_meta.to_csv(preview_csv, index=False)
    print(f"Metadata saved to: {preview_csv}")

if __name__ == "__main__":
    main()
