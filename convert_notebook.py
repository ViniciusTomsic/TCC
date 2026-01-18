import nbformat as nbf
import os

NOTEBOOK_PATH = r'c:\Users\Cliente\Documents\GitHub\TCC\synthetic_signal_sweep.ipynb'

def create_new_notebook():
    nb = nbf.v4.new_notebook()
    
    # Cell 1: Title and Description
    nb.cells.append(nbf.v4.new_markdown_cell("""
# Synthetic Signal Sweep and Visualization

This notebook uses `src/generate_synthetic_data.py` to generate a synthetic dataset of bearing faults and visualizes the results.

The dataset includes:
- Real 'Normal' baseline signals.
- Synthetic fault signals (Inner, Outer, Ball) superimposed on normal baselines.
"""))

    # Cell 2: Imports and Setup
    nb.cells.append(nbf.v4.new_code_cell("""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src is in path
sys.path.append(os.path.abspath('src'))

# Import the generation script
import generate_synthetic_data as gsd

# Configuration for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
"""))

    # Cell 3: Generate Data
    nb.cells.append(nbf.v4.new_code_cell("""
# Run the main generation function
print("Generating synthetic data... This may take a moment.")
df = gsd.main()

print(f"Data generation complete. Shape: {df.shape}")
df.head()
"""))

    # Cell 4: Class Distribution Plot
    nb.cells.append(nbf.v4.new_markdown_cell("## 1. Class Distribution"))
    nb.cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='fault_type', palette='viridis')
plt.title('Distribution of Fault Types in Synthetic Dataset')
plt.xlabel('Fault Type')
plt.ylabel('Count')
plt.show()
"""))

    # Cell 5: Signal Visualization
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 2. Signal Visualization

We will plot a random sample from each class (Normal, Inner, Outer, Ball) to visually verify the signals.
"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""
def plot_signals_by_class(dataframe, num_samples=1):
    fault_types = dataframe['fault_type'].unique()
    
    for f_type in fault_types:
        subset = dataframe[dataframe['fault_type'] == f_type]
        if subset.empty:
            continue
            
        samples = subset.sample(min(num_samples, len(subset)))
        
        for idx, row in samples.iterrows():
            signal = row['signal']
            rpm = row['rpm']
            diam = row['diameter']
            
            plt.figure(figsize=(14, 4))
            plt.plot(signal, label=f'{f_type} (RPM={rpm}, D={diam})', color='blue', linewidth=0.8)
            plt.title(f'Time Domain Signal - {f_type} Fault')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.show()

# Plot 1 random signal from each class
plot_signals_by_class(df, num_samples=1)
"""))

    # Cell 6: FFT Comparison (Optional but helpful)
    nb.cells.append(nbf.v4.new_markdown_cell("## 3. Frequency Domain Comparison (FFT)"))
    nb.cells.append(nbf.v4.new_code_cell("""
def plot_fft_comparison(dataframe):
    # Select one example per fault type for the same RPM if possible
    target_rpm = 1730
    subset = dataframe[dataframe['rpm'] == target_rpm]
    
    if subset.empty:
        print(f"No data for RPM {target_rpm}")
        return

    plt.figure(figsize=(14, 8))
    
    for f_type in ['Normal', 'inner', 'outer', 'ball']:
        class_subset = subset[subset['fault_type'] == f_type]
        if class_subset.empty:
            continue
            
        # Take first sample
        signal = class_subset.iloc[0]['signal']
        
        # FFT
        n = len(signal)
        fs = 12000
        freq = np.fft.rfftfreq(n, d=1/fs)
        fft_val = np.abs(np.fft.rfft(signal)) / n
        
        plt.plot(freq, fft_val, label=f_type, alpha=0.7)

    plt.title(f'FFT Comparison at {target_rpm} RPM')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.xlim(0, 1000) # Zoom in on lower frequencies where bearing faults usually are
    plt.tight_layout()
    plt.show()

plot_fft_comparison(df)
"""))

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Notebook overwritten at {NOTEBOOK_PATH}")

if __name__ == "__main__":
    create_new_notebook()
