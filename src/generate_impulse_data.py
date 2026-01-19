import numpy as np
import pandas as pd
import sys
import os

# Add src to path just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import segment_and_split_data as ssd
import bearing_utils as bu

# =============================================================================
# CONFIGURATION
# =============================================================================

# Parameters for 6205-2 RS bearing (from bearing_utils.py)
# n: number of balls, d: ball diameter, D: pitch diameter
params_drive_end = {
    'n': 9,
    'd': 7.94,       # mm
    'D': 39.05,      # mm
    'phi_graus': 0.0 # contact angle
}

# Reference amplitudes for scaling the impulses
# TODO: Adjust these values based on observed real fault amplitudes
amplitudes_referencia = {
    'Drive End': {
        'Pista Externa': 0.5,
        'Pista Interna': 0.5,
        'Esfera': 0.5,
        'FTF': 0.5 # Included just in case
    }
}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def gerar_dados_sinteticos_treino(
    dicionario_treino,
    TAXA_AMOSTRAL,
    params_drive_end,
    amplitudes_referencia,
    multiplicadores=[2, 5, 10],
    fases_para_adicionar_rad=[0, np.pi/2, np.pi, 3*np.pi/2],
    damping_ratio=0.1,
    duracao_pulso_seg=0.1,
    profundidade_modulacao=0.5
):
    """
    Gera dados sintéticos de falha (versão 'limpa', com menos prints).
    Based on impulse response convolution.
    """

    # --- 1. FUNÇÕES AUXILIARES INTERNAS ---
    
    def calcular_frequencias_rolamento(n, fr, d, D, phi_graus=0.0):
        phi_rad = np.deg2rad(phi_graus)
        termo_comum = (d / D) * np.cos(phi_rad)
        freq_ftf = (fr / 2) * (1 - termo_comum)
        return {
            'Pista Externa': (n * fr / 2) * (1 - termo_comum),
            'Pista Interna': (n * fr / 2) * (1 + termo_comum),
            'Esfera': (D * fr / (2 * d)) * (1 - termo_comum**2),
            'FTF': freq_ftf
        }

    def criar_resposta_impulso(taxa_amostral, tipo_falha, damping, duracao_pulso):
        """
        Cria resposta ao impulso que oscila na frequência natural do anel.
        
        Args:
            taxa_amostral: Taxa de amostragem (Hz)
            tipo_falha: 'Pista Externa', 'Pista Interna' ou 'Esfera'
            damping: Razão de amortecimento
            duracao_pulso: Duração do impulso (segundos)
        """
        # Usar cache de frequências naturais
        if tipo_falha == 'Pista Externa':
            freq_natural = freq_nat_outer
        else:  # Pista Interna ou Esfera
            freq_natural = freq_nat_inner
        
        # Gerar impulso
        n_pontos_pulso = int(duracao_pulso * taxa_amostral)
        if n_pontos_pulso == 0: n_pontos_pulso = 1
        t_pulse = np.linspace(0, duracao_pulso, n_pontos_pulso, endpoint=False)
        A = damping * 2 * np.pi * freq_natural
        omega_d = 2 * np.pi * freq_natural * np.sqrt(1 - damping**2)
        pulso = np.exp(-A * t_pulse) * np.sin(omega_d * t_pulse)
        return pulso
    
    # --- 2. OBTER FREQUÊNCIAS NATURAIS (UMA VEZ APENAS) ---
    df_nat_freq = bu.get_bearing_natural_frequencies()
    df_outer = df_nat_freq[df_nat_freq['Race'] == 'Outer']
    df_inner = df_nat_freq[df_nat_freq['Race'] == 'Inner']
    freq_nat_outer = df_outer.iloc[0]['Freq_Hz']
    freq_nat_inner = df_inner.iloc[0]['Freq_Hz']
    
    print(f"Usando frequências naturais: Outer={freq_nat_outer:.1f} Hz, Inner={freq_nat_inner:.1f} Hz")

    # --- 2. IDENTIFICAÇÃO DOS SEGMENTOS NORMAIS ---
    segmentos_normais_treino = {
        chave: df for chave, df in dicionario_treino.items()
        if df['tipo_falha'].iloc[0] == 'Normal'
    }
    
    if not segmentos_normais_treino:
        print("AVISO: Nenhum segmento normal encontrado no dicionário de treino.")
        return pd.DataFrame()

    # Print de status importante
    print(f"Gerando Sinais... (damp={damping_ratio}, dur={duracao_pulso_seg}s, mod={profundidade_modulacao})")

    # --- 4. GERAÇÃO E COMBINAÇÃO DOS SINAIS ---
    lista_sinais_treino = []
    
    # Check if we should limit the number of normal segments processed
    # The user instruction implies "can decrease parameters we are iterating"
    # But for now, we process all provided normal segments unless the user passes a sliced dict.
    
    for chave_normal, df_normal in segmentos_normais_treino.items():
        sinal_normal_base = df_normal['amplitude'].values
        rpm_atual = df_normal['rotacao_rpm'].iloc[0]
        N_PONTOS = len(sinal_normal_base)
        duracao_s = N_PONTOS / TAXA_AMOSTRAL
        t = np.linspace(0.0, duracao_s, N_PONTOS, endpoint=False)
        fr_hz = rpm_atual / 60
        freqs_teoricas = calcular_frequencias_rolamento(fr=fr_hz, **params_drive_end)
        
        for tipo_falha_sintetica in ['Pista Externa', 'Pista Interna', 'Esfera']:
            freq_teorica = freqs_teoricas[tipo_falha_sintetica]
            amp_ref = amplitudes_referencia['Drive End'][tipo_falha_sintetica]

            # Criar resposta ao impulso específica para este tipo de falha
            resposta_impulso = criar_resposta_impulso(
                TAXA_AMOSTRAL, tipo_falha_sintetica, damping_ratio, duracao_pulso_seg
            )
            # Normalizar
            max_abs_val = np.max(np.abs(resposta_impulso))
            if max_abs_val > 0:
                resposta_impulso /= max_abs_val

            trem_de_impulsos = np.zeros(N_PONTOS)
            periodo_falha_seg = 1.0 / freq_teorica
            ts_segundos = 1.0 / TAXA_AMOSTRAL 
            
            # Create impulse train
            for t_impacto in np.arange(0, duracao_s, periodo_falha_seg):
                idx = int(t_impacto / ts_segundos)
                if idx < N_PONTOS:
                    trem_de_impulsos[idx] = 1.0
            
            sinal_falha_ringing = np.convolve(trem_de_impulsos, resposta_impulso, mode='same')
            
            # Sem modulação (falha gerada apenas na frequência característica)
            sinal_falha_bruto = sinal_falha_ringing
                
            for mult in multiplicadores:
                for fase in fases_para_adicionar_rad:
                    amplitude_final = amp_ref * mult
                    deslocamento_idx = int((fase / (2 * np.pi)) * periodo_falha_seg / ts_segundos)
                    # Use roll to simulate phase shift
                    sinal_falha_sintetico = np.roll(sinal_falha_bruto, deslocamento_idx) * amplitude_final
                    
                    sinal_final_combinado = sinal_normal_base + sinal_falha_sintetico
                    
                    lista_sinais_treino.append({
                        'sinal_final': sinal_final_combinado,
                        'sinal_puro': sinal_falha_sintetico,
                        'tipo_falha_adicionada': tipo_falha_sintetica,
                        'rpm': rpm_atual,
                        'multiplicador_amplitude': mult,
                        'fase_adicionada_rad': fase,
                        'base_normal': chave_normal
                    })

    # --- 5. ADIÇÃO DOS SEGMENTOS NORMAIS ORIGINAIS ---
    for chave_normal, df_normal in segmentos_normais_treino.items():
        # Create zero array for pure signal on normal data
        sinal_zeros = np.zeros_like(df_normal['amplitude'].values)
        
        lista_sinais_treino.append({
            'sinal_final': df_normal['amplitude'].values,
            'sinal_puro': sinal_zeros,
            'tipo_falha_adicionada': 'Normal',
            'rpm': df_normal['rotacao_rpm'].iloc[0],
            'multiplicador_amplitude': 0,
            'fase_adicionada_rad': 0,
            'base_normal': chave_normal
        })

    # --- 6. DATAFRAME INTERMEDIÁRIO ---
    df_sinais_treino = pd.DataFrame(lista_sinais_treino)

    print(f"Geração concluída. Total de {len(df_sinais_treino)} segmentos de treino.")
    
    return df_sinais_treino


if __name__ == "__main__":
    # --- EXECUÇÃO TESTE ---
    print("Iniciando script de geração de sinais por impulso...")
    
    # Reduzindo parâmetros como sugerido pelo usuário para teste rápido
    multipliers_test = [2] # Apenas um multiplicador
    phases_test = [0, np.pi] # Apenas duas fases
    
    # Filtrar um subconjunto de dados normais para não demorar muito no teste
    # Pegamos apenas os primeiros 5 segmentos normais do dicionário carregado
    subset_keys = list(ssd.dicionario_treino.keys())[:5]
    subset_dicionario = {k: ssd.dicionario_treino[k] for k in subset_keys}
    
    print(f"Usando subset de {len(subset_dicionario)} segmentos normais para teste.")
    
    df_resultado = gerar_dados_sinteticos_treino(
        dicionario_treino=subset_dicionario, 
        TAXA_AMOSTRAL=12000, 
        params_drive_end=params_drive_end, 
        amplitudes_referencia=amplitudes_referencia,
        multiplicadores=multipliers_test,
        fases_para_adicionar_rad=phases_test
    )
    
    print(df_resultado['tipo_falha_adicionada'].value_counts())
    print("\nExemplo de dados gerados:")
    print(df_resultado.head())
