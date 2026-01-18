import numpy as np
import pandas as pd
import random
import sys
import os

# Add src to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bearing_utils as bu
# This import triggers data loading and splitting, which might take a moment.
import segment_and_split_data as ssd

# =============================================================================
# CONFIGURATION
# =============================================================================
DIAMETERS = [0.2, 0.5, 1.0, 1.2]
K_VALUES = {
    'inner': [0.05, 0.1, 0.2], 
    'outer': [0.004, 0.008, 0.016], 
    'ball': [0.025, 0.05, 0.1]
}
RPMS = [1730, 1750, 1772, 1797]
FS = 12000  # Sampling rate
NUM_RANDOM_SEGMENTS = 10  # Number of random normal segments to use as baselines

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def synthesize_time_signal(spectrum_df, duration=1.0, fs=FS):
    """
    Reconstructs time domain signal from spectrum df (freq, amp).
    Uses vectorized operations for speed.
    """
    if spectrum_df.empty:
        return np.zeros(int(duration * fs))
        
    t = np.arange(0, duration, 1/fs)
    
    # Extract arrays and reshape for broadcasting
    # Frequencies shape: (N, 1)
    freqs = spectrum_df['Frequency_Hz'].values.reshape(-1, 1)
    
    # Amplitudes shape: (N, 1)
    # NOTE: Using 'Amplitude_Accel_m_s2' as standardized in bearing_utils
    amps = spectrum_df['Amplitude_Accel_m_s2'].values.reshape(-1, 1)
    
    # Time shape: (1, T)
    t_grid = t.reshape(1, -1)
    
    # Cosine argument: 2*pi * f * t
    # Result shape: (N, T)
    cos_components = np.cos(2 * np.pi * freqs * t_grid)
    
    # Weighted sum
    # Result shape: (N, T) -> sum -> (T,)
    signal = np.sum(amps * cos_components, axis=0)
        
    return signal

# =============================================================================
# MAIN GENERATION LOGIC
# =============================================================================

def main():
    print("Starting synthetic data generation...")
    
    generated_data = []

    # 1. Access normal segments from segment_and_split_data
    # ssd.dicionario_treino contains the normal training segments
    # Each item is a DataFrame with 'amplitude' and metadata columns
    
    print(f"Total normal segments available in training set: {len(ssd.dicionario_treino)}")

    # 2. Add ALL normal segments to the final dataset
    # fault_type='Normal', diameter=None, k_val=None
    print("Adding all normal training segments to dataset...")
    for key, df_seg in ssd.dicionario_treino.items():
        signal = df_seg['amplitude'].values
        # Metadata is also available in df_seg but we normalize it for our output format
        rpm = df_seg['rotacao_rpm'].iloc[0] if 'rotacao_rpm' in df_seg else 0
        
        generated_data.append({
            'rpm': rpm,
            'fault_type': 'Normal',
            'diameter': None,
            'k_val': None,
            'signal': signal
        })
    print(f"Added {len(generated_data)} normal segments.")

    # 3. Iterate through RPMs for synthetic generation
    for rpm in RPMS:
        print(f"Processing RPM: {rpm}")
        
        # Filter normal segments for the current RPM
        # We need to keys from dicionario_treino that match the RPM
        # The key format in segment_and_split_data is usually like "{rpm}_Normal_..._seg_{i}"
        # But let's check the 'rotacao_rpm' column in the dataframe just to be sure/cleaner
        
        rpm_segments = []
        for key, df_seg in ssd.dicionario_treino.items():
            if df_seg['rotacao_rpm'].iloc[0] == rpm:
                rpm_segments.append(df_seg['amplitude'].values)
        
        if not rpm_segments:
            print(f"Warning: No normal segments found for RPM {rpm} in training dictionary.")
            continue
            
        # 4. Randomly select segments for baseline
        if len(rpm_segments) > NUM_RANDOM_SEGMENTS:
            baseline_signals = random.sample(rpm_segments, NUM_RANDOM_SEGMENTS)
        else:
            baseline_signals = rpm_segments # Use all if available count is less than target
            
        print(f"  Selected {len(baseline_signals)} baseline segments for synthetic generation.")

        # 5. Generate synthetic faults for each baseline
        segment_size = len(baseline_signals[0]) # Assuming all segments same size, which they are (4096)
        duration_seg = segment_size / FS

        for baseline_sig in baseline_signals:
            for f_type in ['inner', 'outer', 'ball']:
                for k_val in K_VALUES[f_type]:
                    for diam in DIAMETERS:
                        
                        # Calculate Spectrum
                        if f_type == 'inner':
                            spec_df = bu.calcular_espectro_inner_completo(diam, rpm, K=k_val)
                        elif f_type == 'outer':
                            spec_df = bu.calcular_espectro_outer_race(diam, rpm, K=k_val)
                        elif f_type == 'ball':
                            spec_df = bu.calcular_espectro_ball_completo(diam, rpm, K=k_val)
                        
                        # Synthesize Time Domain
                        syn_sig = synthesize_time_signal(spec_df, duration=duration_seg, fs=FS)
                        
                        # Length Check & Alignment
                        if len(syn_sig) > len(baseline_sig):
                            syn_sig = syn_sig[:len(baseline_sig)]
                        elif len(syn_sig) < len(baseline_sig):
                            syn_sig = np.pad(syn_sig, (0, len(baseline_sig) - len(syn_sig)))
                        
                        # SUPERIMPOSITION (Normal + Synthetic Fault)
                        combined_sig = baseline_sig + syn_sig
                        
                        # Add to dataset
                        generated_data.append({
                            'rpm': rpm,
                            'fault_type': f_type,
                            'diameter': diam,
                            'k_val': k_val,
                            'signal': combined_sig
                        })

    # 6. Combine to DataFrame
    final_df = pd.DataFrame(generated_data)
    
    # 7. Final Report
    print("\nGeneration Complete.")
    print(f"Total records in final DataFrame: {len(final_df)}")
    print("\nClass Distribution:")
    print(final_df['fault_type'].value_counts())
    
    print("\nSample Data:")
    print(final_df.head())
    
    # Optional: Save or return? The user requirement says "output final deve conter um dataframe"
    # For a script, usually we assume it runs and maybe saves or just acts as a module.
    # Since this is a standalone script execution request, printing results is good verification.
    
    return final_df

if __name__ == "__main__":
    df = main()
