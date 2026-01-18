import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import segment_and_split_data as ssd
import generate_synthetic_data as gsd
import bearing_utils as bu

def visualize_fft_signals():
    print("--- Preparing FFT Signal Visualization ---")
    
    # 1. Configuration
    # Select one example context
    selected_rpm = 1730
    fs = 12000
    
    print(f"Generating on-the-fly synthetic examples for RPM {selected_rpm}...")
    
    # 2. Get a base normal segment
    base_segment = None
    for key, df in ssd.dicionario_treino.items():
        if df['rotacao_rpm'].iloc[0] == selected_rpm:
            base_segment = df['amplitude'].values
            break
            
    if base_segment is None:
        # Fallback if no specific RPM found (unlikely)
        if len(ssd.dicionario_treino) > 0:
            key = list(ssd.dicionario_treino.keys())[0]
            base_segment = ssd.dicionario_treino[key]['amplitude'].values
            selected_rpm = ssd.dicionario_treino[key]['rotacao_rpm'].iloc[0]
            print(f"Warning: Exact RPM not found, using {selected_rpm} from {key}")
        else:
            print("Error: No normal segments available.")
            return None

    # 3. Define Faults to Visualize
    fault_types = ['Normal', 'inner', 'outer', 'ball']
    
    # Visualization Parameters (matching generate_synthetic_data defaults)
    diam = 0.5 
    k_vals = {'inner': 0.1, 'outer': 0.008, 'ball': 0.05}
    
    # Setup Plot
    rows = len(fault_types)
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), constrained_layout=True)
    
    print("Plotting signals...")
    
    segment_duration = len(base_segment) / fs
    
    for i, f_type in enumerate(fault_types):
        
        # Initialize signals
        signal_pure = np.zeros_like(base_segment)
        signal_final = base_segment.copy() # Start with copies
        
        if f_type != 'Normal':
            # Generate Synthetic Component
            k_val = k_vals.get(f_type, 0.1)
            
            spec_df = pd.DataFrame()
            if f_type == 'inner':
                spec_df = bu.calcular_espectro_inner_completo(diam, selected_rpm, K=k_val)
            elif f_type == 'outer':
                spec_df = bu.calcular_espectro_outer_race(diam, selected_rpm, K=k_val)
            elif f_type == 'ball':
                spec_df = bu.calcular_espectro_ball_completo(diam, selected_rpm, K=k_val)
                
            # Synthesize Time Signal
            syn_sig = gsd.synthesize_time_signal(spec_df, duration=segment_duration, fs=fs)
            
            # Align lengths
            if len(syn_sig) > len(base_segment):
                syn_sig = syn_sig[:len(base_segment)]
            elif len(syn_sig) < len(base_segment):
                syn_sig = np.pad(syn_sig, (0, len(base_segment) - len(syn_sig)))
                
            signal_pure = syn_sig
            signal_final = base_segment + syn_sig
        
        # --- Time Domain Plot (Left) ---
        ax_t = axes[i, 0]
        
        # Time Vector
        n = len(signal_final)
        t = np.linspace(0, n/fs, n)
        
        # Plot Combined/Original
        label_final = "Original Normal" if f_type == 'Normal' else "Combined Signal"
        ax_t.plot(t, signal_final, color='#2c3e50', alpha=0.6, linewidth=0.8, label=label_final)
        
        # Plot Pure Synthetic (if not Normal)
        if f_type != 'Normal':
            ax_t.plot(t, signal_pure, color='#e74c3c', alpha=0.8, linewidth=0.8, label='Pure Synthetic')
            
        ax_t.set_title(f"{f_type.capitalize()} - Time Domain - RPM {selected_rpm}", fontsize=14)
        ax_t.set_xlabel("Time (s)")
        ax_t.set_ylabel("Amplitude")
        ax_t.legend(loc='upper right', fontsize=8)
        ax_t.set_xlim(0, 0.1) # Zoom
        ax_t.grid(True, alpha=0.3)
        
        # --- Frequency Domain Plot (Right) ---
        ax_f = axes[i, 1]
        
        # FFT of Combined
        yf = fft(signal_final)
        xf = fftfreq(n, 1/fs)[:n//2]
        mag = 2.0/n * np.abs(yf[0:n//2])
        
        ax_f.plot(xf, mag, color='#8e44ad', linewidth=0.8, label='Spectrum')
        
        ax_f.set_title(f"{f_type.capitalize()} - FFT", fontsize=14)
        ax_f.set_xlabel("Frequency (Hz)")
        ax_f.set_ylabel("Magnitude")
        ax_f.set_xlim(0, 2000) 
        ax_f.grid(True, alpha=0.3)
    
    print("Plot generation successful!")
    return fig

if __name__ == "__main__":
    visualize_fft_signals()
    plt.show()
