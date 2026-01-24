import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import generate_synthetic_data as gsd
import segment_and_split_data as ssd

def main():
    print("Defining K values...")
    # User specified:
    # 'pista exterena': [0.02, 0.05, 0.1]
    # 'pista interna': [0.05, 0.1, 0.5]
    # 'esfera': [0.02, 0.05, 0.01]
    k_values = {
        'outer': [0.02, 0.05, 0.1],
        'inner': [0.05, 0.1, 0.5],
        'ball': [0.02, 0.05, 0.01]
    }

    print("Generating Training DataFrame (df_treino)...")
    # Uses ssd.dicionario_treino (80% Normals)
    df_treino = gsd.gerar_dados_sinteticos_tandon_df(
        dicionario_treino=ssd.dicionario_treino,
        k_values=k_values,
        incluir_normais_reais=True 
    )
    print(f"df_treino generated with {len(df_treino)} records.")

    print("Generating Test DataFrame (df_teste)...")
    # Uses ssd.dicionario_teste (20% Normals + Real Faults)
    # Note: gerar_dados_sinteticos_tandon_df currently only adds synthetic faults to 'Normal' segments
    # and includes 'Normal' segments. It skips Real Fault segments in the dictionary.
    df_teste = gsd.gerar_dados_sinteticos_tandon_df(
        dicionario_treino=ssd.dicionario_teste, # Passing test dict as 'dicionario_treino' argument
        k_values=k_values,
        incluir_normais_reais=True
    )
    
    # Check if we should manually include real faults in df_teste (which are in ssd.dicionario_teste)
    # The user asked for "df_teste", usually implies the full test set.
    # Since the generator skips real faults, let's append them manually if they exist.
    print("Appending Real Faults to df_teste...")
    real_faults_rows = []
    for key, df_seg in ssd.dicionario_teste.items():
        try:
            val_falha = str(df_seg["tipo_falha"].iloc[0])
            if val_falha == "Normal":
                continue # Already handled by include_normais_reais=True in generator
            
            # This is a real fault
            rpm = int(df_seg["rotacao_rpm"].iloc[0])
            sig = df_seg["amplitude"].values
            
            # Use columns matching the synthetic df
            real_faults_rows.append({
                "rpm": rpm,
                "tipo_falha_adicionada": val_falha, # The actual fault type
                "diametro_falha_mm": np.nan, # Or parse if available, but usually nan for real unless widely known
                "k_val": np.nan,
                "sinal_puro": np.zeros_like(sig), # No pure component separate
                "sinal_final": sig,
                "metodo": "real_fault",
                "base_normal": key
            })
        except Exception as e:
            print(f"Error processing real fault {key}: {e}")
            continue

    if real_faults_rows:
        df_real = pd.DataFrame(real_faults_rows)
        df_teste = pd.concat([df_teste, df_real], ignore_index=True)
        print(f"Added {len(df_real)} real fault records. Total df_teste: {len(df_teste)}")
    else:
        print("No real faults found in dictionary (or separate processing skipped).")

    # Save
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, 'df_treino.pkl')
    test_path = os.path.join(data_dir, 'df_teste.pkl')

    print(f"Saving df_treino to {train_path}...")
    df_treino.to_pickle(train_path)
    
    print(f"Saving df_teste to {test_path}...")
    df_teste.to_pickle(test_path)

    print("\nOverview:")
    print("df_treino:", df_treino['tipo_falha_adicionada'].value_counts())
    print("df_teste:", df_teste['tipo_falha_adicionada'].value_counts())

if __name__ == "__main__":
    main()
