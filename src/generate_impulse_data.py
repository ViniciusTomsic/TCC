import numpy as np
import pandas as pd
import sys
import os
from functools import lru_cache

# Add src to path just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
        'Pista Externa': 0.4,
        'Pista Interna': 0.4,
        'Esfera': 0.4,
        'FTF': 0.5 # Included just in case
    }
}

# =============================================================================
# PUBLIC HELPERS (reutilizáveis em notebooks)
# =============================================================================

@lru_cache(maxsize=1)
def _get_nat_freq_outer_inner():
    """Carrega e separa frequências naturais (cache em memória)."""
    df_nat_freq = bu.get_bearing_natural_frequencies()
    df_nat_freq_outer = df_nat_freq[df_nat_freq["Race"] == "Outer"].reset_index(drop=True)
    df_nat_freq_inner = df_nat_freq[df_nat_freq["Race"] == "Inner"].reset_index(drop=True)
    return df_nat_freq_outer, df_nat_freq_inner


def calcular_frequencias_rolamento(n, fr, d, D, phi_graus=0.0):
    """Frequências características teóricas (Hz)."""
    phi_rad = np.deg2rad(phi_graus)
    termo_comum = (d / D) * np.cos(phi_rad)
    freq_ftf = (fr / 2) * (1 - termo_comum)
    return {
        "Pista Externa": (n * fr / 2) * (1 - termo_comum),
        "Pista Interna": (n * fr / 2) * (1 + termo_comum),
        "Esfera": (D * fr / (2 * d)) * (1 - termo_comum**2),
        "FTF": freq_ftf,
    }


def criar_resposta_impulso(
    taxa_amostral: float,
    tipo_falha: str,
    damping: float,
    duracao_pulso: float,
    num_modos: int = 6,
):
    """
    Cria resposta ao impulso como soma ponderada de múltiplas frequências naturais.

    Observação: esta função é usada no `general_sam_analysis.ipynb` para evitar
    recriar manualmente o sinal no notebook.
    """
    df_nat_freq_outer, df_nat_freq_inner = _get_nat_freq_outer_inner()

    # Selecionar DataFrame de frequências naturais baseado no tipo de falha
    if tipo_falha == "Pista Externa":
        df_race = df_nat_freq_outer
    else:  # Pista Interna ou Esfera
        df_race = df_nat_freq_inner

    modos = df_race.head(num_modos)

    n_pontos_pulso = int(duracao_pulso * taxa_amostral)
    if n_pontos_pulso <= 0:
        n_pontos_pulso = 1
    t_pulse = np.linspace(0, duracao_pulso, n_pontos_pulso, endpoint=False)

    pulso_total = np.zeros(n_pontos_pulso)
    for _, modo in modos.iterrows():
        freq_natural = modo["Freq_Hz"]
        mass = modo["Mass_kg"]
        omega_n = 2 * np.pi * freq_natural
        if omega_n <= 0:
            continue

        # Receptância modal (inversamente proporcional à rigidez modal)
        receptance = 1.0 / (mass * omega_n**2)
        A = damping * omega_n
        omega_d = omega_n * np.sqrt(1 - damping**2)
        pulso_modo = receptance * np.exp(-A * t_pulse) * np.sin(omega_d * t_pulse)
        pulso_total += pulso_modo

    return pulso_total


def gerar_sinal_impulso_completo(
    *,
    fs: float,
    duration_points: int,
    defect_freq_hz: float,
    tipo_falha_str: str,
    damping: float = 0.1,
    duracao_pulso: float = 0.02,
    num_modos: int = 6,
):
    """
    Gera sinal sintético 'puro' via convolução:
    trem de impulsos (defect_freq_hz) * resposta ao impulso (modos naturais).
    """
    duration_sec = duration_points / fs

    # 1) Trem de impulsos
    trem = np.zeros(duration_points)
    if defect_freq_hz <= 0:
        return trem

    periodo_s = 1.0 / defect_freq_hz
    ts = 1.0 / fs
    for t_imp in np.arange(0, duration_sec, periodo_s):
        idx = int(t_imp / ts)
        if idx < duration_points:
            trem[idx] = 1.0

    # 2) Resposta ao impulso
    resp_imp = criar_resposta_impulso(
        taxa_amostral=fs,
        tipo_falha=tipo_falha_str,
        damping=damping,
        duracao_pulso=duracao_pulso,
        num_modos=num_modos,
    )
    max_abs = float(np.max(np.abs(resp_imp))) if len(resp_imp) else 0.0
    if max_abs > 0:
        resp_imp = resp_imp / max_abs

    # 3) Convolução
    return np.convolve(trem, resp_imp, mode="same")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def gerar_dados_sinteticos_treino(
    dicionario_treino,
    TAXA_AMOSTRAL,
    params_drive_end,
    amplitudes_referencia,
    multiplicadores=[1],
    fases_para_adicionar_rad=[0, np.pi/2, np.pi, 3*np.pi/2],
    damping_ratio=0.1,
    duracao_pulso_seg=0.1,
    profundidade_modulacao=0.5
):
    """
    Gera dados sintéticos de falha (versão 'limpa', com menos prints).
    Based on impulse response convolution.
    """

    # --- 1. Frequências naturais (log apenas) ---
    df_nat_freq_outer, df_nat_freq_inner = _get_nat_freq_outer_inner()
    if len(df_nat_freq_outer) > 0 and len(df_nat_freq_inner) > 0:
        freq_nat_outer_first = df_nat_freq_outer.iloc[0]["Freq_Hz"]
        freq_nat_inner_first = df_nat_freq_inner.iloc[0]["Freq_Hz"]
        print(
            f"Usando {len(df_nat_freq_outer)} modos naturais para Outer Race (primeiro: {freq_nat_outer_first:.1f} Hz)"
        )
        print(
            f"Usando {len(df_nat_freq_inner)} modos naturais para Inner Race (primeiro: {freq_nat_inner_first:.1f} Hz)"
        )

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
    import segment_and_split_data as ssd

    # --- EXECUÇÃO TESTE ---
    print("Iniciando script de geração de sinais por impulso...")
    
    # Reduzindo parâmetros como sugerido pelo usuário para teste rápido
    multipliers_test = [1] # Apenas um multiplicador
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
