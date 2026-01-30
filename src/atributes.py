# %% [markdown]
# ## Importação dos arquivos e geração dos segmentos

# %%

# %%
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

try:
    from src.general_sam_analysis_utils import apply_hanning_window, get_mag_spectrum, apply_lowpass_filter, limit_spectrum_frequency
except ImportError:
    from general_sam_analysis_utils import apply_hanning_window, get_mag_spectrum, apply_lowpass_filter, limit_spectrum_frequency

# %% [markdown]
# ## Extração de atributos

# %%
from scipy.stats import skew

# =============================================================================
# BLOCO 3 (Refatorado): FUNÇÃO DE EXTRAÇÃO DE FEATURES
# =============================================================================

def extrair_features_treino_teste(
    df_sinais_treino: pd.DataFrame, 
    dicionario_teste: Dict[str, pd.DataFrame], 
    TAXA_AMOSTRAL: float,
    min_freq_pico: float = 50, 
    max_freq_pico: float = 200,
    min_freq_fft: float = 90,
    max_freq_fft: float = 1200
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extrai atributos (versão 'limpa', com menos prints).
    
    Args:
        df_sinais_treino: DataFrame contendo os sinais de treino e metadados.
        dicionario_teste: Dicionário onde cada chave mapeia para um DataFrame com o sinal de teste.
        TAXA_AMOSTRAL: Frequência de amostragem dos sinais.
        min_freq_pico: Frequência mínima para busca de pico.
        max_freq_pico: Frequência máxima para busca de pico.
        min_freq_fft: Frequência mínima para cálculo global dos atributos espectrais.
        max_freq_fft: Frequência máxima para cálculo global dos atributos espectrais.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames de features de treino e teste.
    """

# =============================================================================
# BLOCO 3 (Refatorado): FUNÇÕES DE EXTRAÇÃO DE FEATURES
# =============================================================================

def calcular_tf2_std(sinal): return np.std(sinal)
def calcular_tf3_rms(sinal): return np.sqrt(np.mean(sinal**2))
def calcular_tf4_fator_forma(sinal):
    rms = calcular_tf3_rms(sinal)
    media_abs = np.mean(np.abs(sinal))
    return rms / media_abs if media_abs != 0 else 0

def calcular_features_frequencia(sinal, taxa_amostral, min_f_fft, max_f_fft):
    N = len(sinal)
    if N == 0: return 0, 0, 0
    
    # NÃO aplica filtro passa-baixa nem janela de Hanning
    espectro = get_mag_spectrum(sinal)
    
    freqs = np.fft.fftfreq(N, 1 / taxa_amostral)[:N//2]
    
    # NÃO limita o espectro para o cálculo dos momentos (usa o full range disponível ou definido externamente se quisesse, 
    # mas o pedido foi remover limites. Se mantivermos os argumentos na assinatura por compatibilidade, apenas não usaremos
    # para 'cortar' o vetor, ou usaremos para selecionar a faixa de interesse DO PONTO DE VISTA DE FEATURES, 
    # mas o user disse "quero remover processamento de sinais e limites".
    # Vou assumir que ele quer calcular sobre TODO o espectro ou que os limites passados são irrelevantes agora.
    # Mas para ser seguro e seguir a risca "remover limites":
    
    soma_espectro = np.sum(espectro)
    if soma_espectro == 0: return 0, 0, 0
    ff2_freq_central = np.sum(freqs * espectro) / soma_espectro
    ff3_rms_freq = np.sqrt(np.sum((freqs**2) * espectro) / soma_espectro)
    ff4_std_freq = np.sqrt(np.sum(((freqs - ff2_freq_central)**2) * espectro) / soma_espectro)
    numerador_ff5 = np.sum(((freqs - ff2_freq_central)**3) * espectro) / soma_espectro
    ff5_assimetria = numerador_ff5 / (ff4_std_freq**3) if ff4_std_freq != 0 else 0
    return ff2_freq_central, ff3_rms_freq, ff5_assimetria

def calcular_freq_pico_range(sinal, taxa_amostral, min_freq, max_freq):
    N = len(sinal)
    if N == 0: return 0
    
    # NÃO aplica filtro passa-baixa nem janela de Hanning
    espectro = get_mag_spectrum(sinal)
    
    freqs = np.fft.fftfreq(N, 1 / taxa_amostral)[:N//2]
    
    # Aqui o range_mask é funcionalidade de "busca de pico em banda específica" (ex: achar pico em 50-200Hz).
    # O user pediu remover "limites". Vou remover a filtragem do SINAL. 
    # A busca do pico em range especifico é feature util. Vou MANTER a busca no range, pois "limit_spectrum_frequency" 
    # refere-se mais a cortar o sinal todo. Se o user quiser o pico global, ele passaria 0-Nyquist.
    # Mas o pedido "remover processamento de sinais e limites" é forte.
    # O código original aplicava filtros ANTES.
    # Agora estamos usando o sinal puro.
    
    range_mask = (freqs >= min_freq) & (freqs <= max_freq)
    freqs_filtradas = freqs[range_mask]
    espectro_filtrado = espectro[range_mask]
    if len(espectro_filtrado) == 0: return 0 
    indice_pico = np.argmax(espectro_filtrado)
    freq_pico = freqs_filtradas[indice_pico]
    return freq_pico

def extrair_todas_features(sinal, taxa_amostral, min_f, max_f, min_f_fft, max_f_fft):
    tf2 = calcular_tf2_std(sinal)
    tf3 = calcular_tf3_rms(sinal)
    tf4 = calcular_tf4_fator_forma(sinal)
    ff2, ff3, ff5 = calcular_features_frequencia(sinal, taxa_amostral, min_f_fft, max_f_fft)
    ff_pico_range = calcular_freq_pico_range(sinal, taxa_amostral, min_freq=min_f, max_freq=max_f)
    return {
        'TF2_std': tf2, 'TF3_rms': tf3, 'TF4_fator_forma': tf4,
        'FF2_freq_central': ff2, 'FF3_rms_freq': ff3, 'FF5_assimetria_espectral': ff5,
        'FF_pico_50_200Hz': ff_pico_range
    }

def extrair_features_from_df(
    df: pd.DataFrame, 
    TAXA_AMOSTRAL: float,
    min_freq_pico: float = 50, 
    max_freq_pico: float = 200,
    min_freq_fft: float = 90,
    max_freq_fft: float = 1200,
    signal_col: str = 'sinal_final'
) -> pd.DataFrame:
    """
    Extrai features de um DataFrame contendo sinais.
    """
    lista_de_features = []
    
    if not df.empty:
        for linha in df.itertuples():
            sinal = getattr(linha, signal_col, None)
            if sinal is None: 
                 continue

            features = extrair_todas_features(
                sinal, TAXA_AMOSTRAL, min_freq_pico, max_freq_pico, min_freq_fft, max_freq_fft
            )
            
            # Copia metadados existentes da linha
            # Excluindo o sinal para economizar memória no DF de features se desejado,
            # ou mantendo. O usuário pediu "sinal final de treino e teste com os atributos calculados",
            # então idealmente deveríamos manter tudo ou juntar depois.
            # Aqui vamos criar um dicionário com os atributos e depois o usuário pode juntar ou
            # podemos adicionar os metadados aqui.
            
            # Preserving common metadata
            features['tipo_falha_adicionada'] = getattr(linha, 'tipo_falha_adicionada', 'Desconhecido')
            features['rpm'] = getattr(linha, 'rpm', 0)
            features['k_val'] = getattr(linha, 'k_val', getattr(linha, 'multiplicador_amplitude', np.nan))
            features['base_normal'] = getattr(linha, 'base_normal', 'Desconhecido')
            features['diametro_falha_mm'] = getattr(linha, 'diametro_falha_mm', np.nan)
            
            # Se houver outras colunas relevantes, adicione aqui.
            # Para garantir que podemos mergir de volta ou identificar, seria bom ter um ID, 
            # mas vamos assumir ordem preservada.
            
            lista_de_features.append(features)

    return pd.DataFrame(lista_de_features)

def extrair_features_treino_teste(
    df_sinais_treino: pd.DataFrame, 
    dicionario_teste: Dict[str, pd.DataFrame], 
    TAXA_AMOSTRAL: float,
    min_freq_pico: float = 50, 
    max_freq_pico: float = 200,
    min_freq_fft: float = 90,
    max_freq_fft: float = 1200
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extrai atributos (versão 'limpa', mantida para compatibilidade).
    """


    # =============================================================================
    # --- 2. PROCESSAMENTO DO CONJUNTO DE TREINO ---
    # =============================================================================
    print("Extraindo features de TREINO...")
    df_treino = extrair_features_from_df(
        df_sinais_treino, 
        TAXA_AMOSTRAL, 
        min_freq_pico, 
        max_freq_pico, 
        min_freq_fft, 
        max_freq_fft
    )

    # =============================================================================
    # --- 3. PROCESSAMENTO DO CONJUNTO DE TESTE ---
    # =============================================================================
    print("Extraindo features de TESTE...")
    lista_de_features_teste = []
    
    if dicionario_teste:
        for chave, df_segmento in dicionario_teste.items():
            if df_segmento.empty:
                continue
                
            sinal = df_segmento['amplitude'].values
            features = extrair_todas_features(
                sinal, TAXA_AMOSTRAL, min_freq_pico, max_freq_pico, min_freq_fft, max_freq_fft
            )
            features['tipo_falha_adicionada'] = df_segmento['tipo_falha'].iloc[0]
            features['rpm'] = df_segmento['rotacao_rpm'].iloc[0]
            features['arquivo_origem'] = getattr(df_segmento, 'arquivo_origem', 'Desconhecido') # pode não existir em todos os casos
            
            # Adiciona informações extras se disponíveis no dicionário (geralmente não estão no formato padrão do dicionario_teste, 
            # mas garante consistência se mudarmos)
            # Para teste real, não temos k_val simulado, então fica NaN ou 0
            features['k_val'] = np.nan 
            
            lista_de_features_teste.append(features)

    df_teste = pd.DataFrame(lista_de_features_teste)
    print(f"Extração concluída. {len(df_treino)} amostras de treino, {len(df_teste)} amostras de teste.") 

    return df_treino, df_teste


