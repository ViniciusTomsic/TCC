import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Adicione o caminho 'src' ao path se necessário, caso esteja rodando do notebook na raiz
# sys.path.append(os.path.abspath('src'))

# Importações dos seus scripts
import segment_and_split_data as ssd
from generate_impulse_data import gerar_dados_sinteticos_treino, params_drive_end, amplitudes_referencia

def visualizar_sinais_impulso(selected_rpm=1730, save_path=None):
    print("--- Preparando Visualização ---")
    
    # 1. Selecionar um subset com o RPM especificado
    dicionario_subset = {}
    for k, df in ssd.dicionario_treino.items():
        if df['rotacao_rpm'].iloc[0] == selected_rpm:
            dicionario_subset[k] = df
            break
    
    # Se não encontrar, usar o primeiro disponível
    if not dicionario_subset:
        keys_subset = list(ssd.dicionario_treino.keys())[:1]
        dicionario_subset = {k: ssd.dicionario_treino[k] for k in keys_subset}
        selected_rpm = dicionario_subset[list(dicionario_subset.keys())[0]]['rotacao_rpm'].iloc[0]
        print(f"Warning: RPM exato não encontrado, usando RPM {selected_rpm}")
    
    print(f"Usando segmento base com RPM {selected_rpm}: {list(dicionario_subset.keys())[0]}")

    # 2. Gerar dados sintéticos
    # Usando multiplicador alto (e.g. 5 ou 10) para destacar bem os impulsos no plot
    print("Gerando sinais sintéticos...")
    df_viz = gerar_dados_sinteticos_treino(
        dicionario_treino=dicionario_subset,
        TAXA_AMOSTRAL=12000,
        params_drive_end=params_drive_end,
        amplitudes_referencia=amplitudes_referencia,
        multiplicadores=[1],  # Amplitude x5 para visualização clara
        fases_para_adicionar_rad=[0], # Apenas fase 0
        damping_ratio=0.1,
        duracao_pulso_seg=0.1, # Pulso curto
        profundidade_modulacao=0.5
    )

    # 3. Plotagem
    tipos_falha = ['Normal', 'Pista Externa', 'Pista Interna', 'Esfera']
    cols = 2
    rows = len(tipos_falha)
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), constrained_layout=True)
    
    fs = 12000
    
    for i, tipo in enumerate(tipos_falha):
        # Filtrar o primeiro exemplo deste tipo
        subset = df_viz[df_viz['tipo_falha_adicionada'] == tipo]
        if subset.empty:
            continue
            
        row_data = subset.iloc[0]
        sinal_final = row_data['sinal_final']
        sinal_puro = row_data['sinal_puro']
        rpm = row_data['rpm']
        
        # Vetor de tempo
        n = len(sinal_final)
        t = np.linspace(0, n/fs, n)
        
        # --- Plot no Tempo (Esquerda) ---
        ax_t = axes[i, 0]
        # Plot Combined
        ax_t.plot(t, sinal_final, color='#2c3e50', alpha=0.6, linewidth=0.8, label='Combinado')
        
        # Plot Pure Synthetic (if not Normal)
        if tipo != 'Normal':
            ax_t.plot(t, sinal_puro, color='#e74c3c', alpha=0.8, linewidth=0.8, label='Sintético Puro')
            
        ax_t.legend(loc='upper right', fontsize=8)
        
        # Destaque se não for normal
        if tipo != 'Normal':
            ax_t.set_title(f"{tipo} (Sintético) - Tempo - RPM {rpm:.0f}", fontsize=14, color='#e74c3c')
        else:
            ax_t.set_title(f"{tipo} (Original) - Tempo - RPM {rpm:.0f}", fontsize=14, color='#27ae60')
            
        ax_t.set_xlabel("Tempo (s)")
        ax_t.set_ylabel("Amplitude")
        ax_t.set_xlim(0, 0.2) # Zoom nos primeiros 200ms
        ax_t.grid(True, alpha=0.3)
        
        # --- Plot na Frequência (Direita) ---
        ax_f = axes[i, 1]
        
        # Cálculo da FFT
        yf = fft(sinal_final)
        xf = fftfreq(n, 1/fs)[:n//2]
        mag = 2.0/n * np.abs(yf[0:n//2])
        
        ax_f.plot(xf, mag, color='#8e44ad', linewidth=0.8)
        ax_f.set_title(f"{tipo} - Espectro (FFT)", fontsize=14)
        ax_f.set_xlabel("Frequência (Hz)")
        ax_f.set_ylabel("Magnitude")
        ax_f.set_xlim(0, 4000) # Foco até 4kHz para ver ressonâncias
        ax_f.grid(True, alpha=0.3)
        
        # Validar frequencias teóricas
        if tipo != 'Normal':
             # Recalcular frequencias esperadas para desenhar linhas verticais (cálculo rápido local)
            fr = rpm/60
            # Aproximação rápida baseada no script anterior
            f_expected = 0
            if tipo == 'Pista Externa': f_expected = 3.58 * fr * 9 # BPFO approx
            # Adicione markers se desejar, mas o visual já ajuda
            
    print("Plot gerado com sucesso!")
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig

# Se rodar este script diretamente pelo terminal:
if __name__ == "__main__":
    fig = visualizar_sinais_impulso(selected_rpm=1730, save_path="impulse_signals_visualization.png")
    print("Visualization complete!")

