import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from atributes import extrair_todas_features

def add_features_to_dataframe(df, taxa_amostral=12000):
    """
    Calculates features for each signal in the dataframe and adds them as new columns.
    """
    if df.empty:
        return df

    print(f"Processing {len(df)} records...")
    
    # Define parameters for feature extraction
    # Using defaults from atributes.py but modifying for test:
    # Keep lowpass (1200Hz) but remove lower limits
    min_freq_pico = 0  
    max_freq_pico = 3000
    min_freq_fft = 0   
    max_freq_fft = 3000
    
    # Function to apply row-wise
    def _extract_wrapper(row):
        sinal = row['sinal_final']
        if sinal is None or len(sinal) == 0:
             # Return a dict of NaNs with correct keys if possible, or handle later
             # For now, let's assume valid signals or return empty dict (normalization will make NaNs)
             return {}
        
        return extrair_todas_features(
            sinal, 
            taxa_amostral, 
            min_freq_pico, 
            max_freq_pico, 
            min_freq_fft, 
            max_freq_fft
        )

    # Apply extraction
    # We use apply on the Series of signals for speed/simplicity
    features_series = df.apply(_extract_wrapper, axis=1)
    
    # Normalize dictionary to columns
    features_df = pd.json_normalize(features_series)
    
    # Assign index to match original df to ensure correct alignment
    features_df.index = df.index
    
    # Concatenate features with original dataframe
    # We use join to be safe with indices
    df_result = df.join(features_df)
    
    return df_result

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    train_path = os.path.join(data_dir, 'df_treino.pkl')
    test_path = os.path.join(data_dir, 'df_teste.pkl')
    
    output_train_path = os.path.join(data_dir, 'df_treino_com_atributos.pkl')
    output_test_path = os.path.join(data_dir, 'df_teste_com_atributos.pkl')

    # Load DataFrames
    print(f"Loading {train_path}...")
    try:
        df_treino = pd.read_pickle(train_path)
    except FileNotFoundError:
        print(f"Error: {train_path} not found. Please run generate_train_test_datasets.py first.")
        return

    print(f"Loading {test_path}...")
    try:
        df_teste = pd.read_pickle(test_path)
    except FileNotFoundError:
        print(f"Error: {test_path} not found. Please run generate_train_test_datasets.py first.")
        return

    # Process Train
    print("\n--- Processing Training Data ---")
    df_treino_final = add_features_to_dataframe(df_treino, taxa_amostral=12000)
    
    # Drop signal columns as requested
    cols_to_drop = ['sinal_final', 'sinal_puro']
    print(f"Dropping columns: {cols_to_drop}")
    df_treino_final.drop(columns=[c for c in cols_to_drop if c in df_treino_final.columns], inplace=True)
    
    print(f"Saving training data with attributes to {output_train_path}...")
    df_treino_final.to_pickle(output_train_path)
    
    # Process Test
    print("\n--- Processing Test Data ---")
    df_teste_final = add_features_to_dataframe(df_teste, taxa_amostral=12000)
    
    # Drop signal columns as requested
    df_teste_final.drop(columns=[c for c in cols_to_drop if c in df_teste_final.columns], inplace=True)
    
    print(f"Saving test data with attributes to {output_test_path}...")
    df_teste_final.to_pickle(output_test_path)

    print("\nDone!")
    print("Columns in new dataframes:")
    print(df_treino_final.columns.tolist())

if __name__ == "__main__":
    main()
