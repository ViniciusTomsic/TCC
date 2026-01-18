# %% [markdown]
# ## Importação dos arquivos e geração dos segmentos

# %%
import os
import pandas as pd
import numpy as np
import pprint

# =============================================================================
# BLOCO 1: CONFIGURAÇÃO, CARREGAMENTO, DIVISÃO (80/20) E SEGMENTAÇÃO (CORRIGIDO)
# =============================================================================

# --- 1. CONFIGURAÇÕES GERAIS ---
caminho_raiz = r'C:\Users\vinic\OneDrive\Documentos\Graduação\TG\Dataset' # IMPORTANTE: Verifique se este caminho está correto
params_drive_end = {'n': 9, 'd': 0.3126, 'D': 1.537, 'phi_graus': 0.0}
TAXA_AMOSTRAL = 12000

# Dicionários de mapeamento
mapa_tipo_falha = {'IR': 'Pista Interna', 'B': 'Esfera', 'OR': 'Pista Externa', 'Normal': 'Normal'}
mapa_diametro_falha = {'7': '0.007"', '14': '0.014"', '21': '0.021"'}

# --- PARÂMETROS DE SEGMENTAÇÃO ---
tamanho_segmento = 4096
sobreposicao_percentual = 0.3
passo = int(tamanho_segmento * (1 - sobreposicao_percentual))

# --- 2. CARREGAMENTO, DIVISÃO E PROCESSAMENTO ---
dicionario_treino = {} # Dicionário para 80% dos dados normais
dicionario_teste = {} # Dicionário para 20% normais + 100% falhas reais

print(f"Iniciando a leitura e segmentação dos arquivos em '{caminho_raiz}'...")
print("Dados normais serão divididos (80% treino / 20% teste).")
print("Dados de falha real irão 100% para o teste.")

# Função auxiliar para segmentar um sinal e adicionar ao dicionário
def segmentar_e_adicionar(sinal, metadados, dicionario_alvo, chave_base):
    # Verifica se o sinal é longo o suficiente para pelo menos um segmento
    if len(sinal) < tamanho_segmento:
        # print(f"Aviso: Sinal da base '{chave_base}' muito curto ({len(sinal)} amostras) para gerar segmentos. Ignorando.")
        return 0

    num_segmentos_criados = 0
    for i, inicio in enumerate(range(0, len(sinal) - tamanho_segmento + 1, passo)):
        segmento = sinal[inicio : inicio + tamanho_segmento]
        df_segmento = pd.DataFrame({'amplitude': segmento})

        # Adiciona metadados
        df_segmento['arquivo_origem'] = metadados['nome_arquivo']
        df_segmento['rotacao_rpm'] = metadados['rpm']
        df_segmento['tipo_falha'] = metadados['tipo_falha']
        df_segmento['diametro_falha'] = metadados['diametro_falha']
        df_segmento['local_sensor'] = 'Drive End'

        chave_segmento = f"{chave_base}_seg_{i}"
        dicionario_alvo[chave_segmento] = df_segmento
        num_segmentos_criados += 1
    return num_segmentos_criados

# Loop principal pelos arquivos
for pasta_atual, _, arquivos in os.walk(caminho_raiz):
    for nome_arquivo in arquivos:
        # Processar apenas arquivos .npz
        if nome_arquivo.endswith('.npz'):
            caminho_completo = os.path.join(pasta_atual, nome_arquivo)

            # Decodificação de metadados
            nome_sem_ext = nome_arquivo.replace('.npz', '')
            partes = nome_sem_ext.split('_')
            rpm_str = partes[0]
            is_normal = 'Normal' in nome_arquivo

            metadados = {
                'nome_arquivo': nome_arquivo,
                'rpm': int(rpm_str) if rpm_str.isdigit() else 0,
                'tipo_falha': 'Normal' if is_normal else mapa_tipo_falha.get(partes[1].split('@')[0], 'Desconhecido'),
                'diametro_falha': 'N/A' if is_normal else mapa_diametro_falha.get(partes[2], 'Desconhecido')
            }

            try:
                dados_npz = np.load(caminho_completo)
                sensor_cod = 'DE' # Foco apenas no Drive End, como no seu código original

                if sensor_cod in dados_npz.files:
                    sinal_completo = dados_npz[sensor_cod].ravel()

                    if is_normal:
                        # DIVIDE O SINAL NORMAL EM 80/20
                        ponto_corte = int(len(sinal_completo) * 0.8)
                        sinal_treino = sinal_completo[:ponto_corte]
                        sinal_teste = sinal_completo[ponto_corte:]

                        chave_base_normal = f"{nome_sem_ext}_{sensor_cod}"
                        segmentar_e_adicionar(sinal_treino, metadados, dicionario_treino, f"{chave_base_normal}_treino")
                        segmentar_e_adicionar(sinal_teste, metadados, dicionario_teste, f"{chave_base_normal}_teste")

                    else:
                        # Sinais com falha (REAIS) vão inteiramente para o TESTE
                        # Lógica de chave para arquivos de falha (igual ao seu original)
                        partes_chave = nome_sem_ext.split('_')
                        partes_chave[-1] = partes_chave[-1].rstrip('0123456789')
                        chave_base_falha = "_".join(partes_chave)
                        
                        # =================================================================
                        # MUDANÇA PRINCIPAL AQUI: Envia falhas reais para o dicionario_teste
                        # =================================================================
                        segmentar_e_adicionar(sinal_completo, metadados, dicionario_teste, chave_base_falha)

            except Exception as e:
                print(f"Erro ao processar o arquivo {nome_arquivo}: {e}")

# --- Relatório Final (Atualizado para refletir a nova lógica) ---
print("\n--- Processo Concluído! ---")
print(f"Total de segmentos de TREINO (APENAS 80% normais): {len(dicionario_treino)}")
print(f"Total de segmentos de TESTE (falhas reais + 20% normais): {len(dicionario_teste)}")

if not dicionario_teste:
    print("\nAVISO: O dicionário de teste está vazio. Verifique se os arquivos 'Normal' existem e se os sinais são longos o suficiente.")

if dicionario_treino:
    # Garante que dicionário não está vazio antes de tentar acessar
    if len(dicionario_treino) > 0:
        chave_exemplo_treino = list(dicionario_treino.keys())[0]
        print(f"\nExemplo de um segmento de TREINO (chave: '{chave_exemplo_treino}'):")
        print(dicionario_treino[chave_exemplo_treino].head())
    else:
        print("\nO dicionário de TREINO está vazio.")

if dicionario_teste:
     # Garante que dicionário não está vazio antes de tentar acessar
    if len(dicionario_teste) > 0:
        chave_exemplo_teste = list(dicionario_teste.keys())[0]
        print(f"\nExemplo de um segmento de TESTE (chave: '{chave_exemplo_teste}'):")
        print(dicionario_teste[chave_exemplo_teste].head())
    else:
        print("\nO dicionário de TESTE está vazio.")

# %% [markdown]
# ## Gera os sinais sintéticos

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# BLOCO 2 (Refatorado): FUNÇÃO DE GERAÇÃO DE DADOS SINTÉTICOS
# =============================================================================

def gerar_dados_sinteticos_treino(
    dicionario_treino,
    TAXA_AMOSTRAL,
    params_drive_end,
    amplitudes_referencia,
    multiplicadores=[2, 5, 10],
    fases_para_adicionar_rad=[0, np.pi/2, np.pi, 3*np.pi/2],
    freq_natural_hz=3278,
    damping_ratio=0.1,
    duracao_pulso_seg=0.1,
    profundidade_modulacao=0.5
):
    """
    Gera dados sintéticos de falha (versão 'limpa', com menos prints).
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

    def criar_resposta_impulso(taxa_amostral, freq_natural, damping, duracao_pulso):
        n_pontos_pulso = int(duracao_pulso * taxa_amostral)
        if n_pontos_pulso == 0: n_pontos_pulso = 1
        t_pulse = np.linspace(0, duracao_pulso, n_pontos_pulso, endpoint=False)
        A = damping * 2 * np.pi * freq_natural
        omega_d = 2 * np.pi * freq_natural * np.sqrt(1 - damping**2)
        pulso = np.exp(-A * t_pulse) * np.sin(omega_d * t_pulse)
        return pulso

    # --- 2. IDENTIFICAÇÃO DOS SEGMENTOS NORMAIS ---
    segmentos_normais_treino = {
        chave: df for chave, df in dicionario_treino.items()
        if df['tipo_falha'].iloc[0] == 'Normal'
    }
    # print(f"Usando {len(segmentos_normais_treino)} segmentos normais de TREINO...") # <-- REMOVIDO

    # --- 3. CRIAÇÃO DA RESPOSTA AO IMPULSO ---
    resposta_impulso_unitaria = criar_resposta_impulso(
        TAXA_AMOSTRAL, freq_natural_hz, damping_ratio, duracao_pulso_seg
    )
    max_abs_val = np.max(np.abs(resposta_impulso_unitaria))
    if max_abs_val > 0:
        resposta_impulso_unitaria /= max_abs_val

    # Print de status importante (mantido)
    print(f"Gerando Sinais... (fn={freq_natural_hz}Hz, damp={damping_ratio}, dur={duracao_pulso_seg}s, mod={profundidade_modulacao})")

    # --- 4. GERAÇÃO E COMBINAÇÃO DOS SINAIS ---
    lista_sinais_treino = []
    
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

            trem_de_impulsos = np.zeros(N_PONTOS)
            periodo_falha_seg = 1.0 / freq_teorica
            ts_segundos = 1.0 / TAXA_AMOSTRAL 
            for t_impacto in np.arange(0, duracao_s, periodo_falha_seg):
                idx = int(t_impacto / ts_segundos)
                if idx < N_PONTOS:
                    trem_de_impulsos[idx] = 1.0
            
            sinal_falha_ringing = np.convolve(trem_de_impulsos, resposta_impulso_unitaria, mode='same')
            
            if tipo_falha_sintetica == 'Pista Externa':
                sinal_falha_bruto = sinal_falha_ringing
            elif tipo_falha_sintetica == 'Pista Interna':
                modulador_fr = (1 + profundidade_modulacao * np.sin(2 * np.pi * fr_hz * t))
                sinal_falha_bruto = sinal_falha_ringing * modulador_fr
            elif tipo_falha_sintetica == 'Esfera':
                freq_ftf = freqs_teoricas['FTF']
                modulador_ftf = (1 + profundidade_modulacao * np.sin(2 * np.pi * freq_ftf * t))
                sinal_falha_bruto = sinal_falha_ringing * modulador_ftf
                
            for mult in multiplicadores:
                for fase in fases_para_adicionar_rad:
                    amplitude_final = amp_ref * mult
                    deslocamento_idx = int((fase / (2 * np.pi)) * periodo_falha_seg / ts_segundos)
                    sinal_falha_sintetico = np.roll(sinal_falha_bruto, deslocamento_idx) * amplitude_final
                    sinal_final_combinado = sinal_normal_base + sinal_falha_sintetico
                    lista_sinais_treino.append({
                        'sinal_final': sinal_final_combinado,
                        'tipo_falha_adicionada': tipo_falha_sintetica,
                        'rpm': rpm_atual,
                        'multiplicador_amplitude': mult,
                        'fase_adicionada_rad': fase,
                        'base_normal': chave_normal
                    })

    # --- 5. ADIÇÃO DOS SEGMENTOS NORMAIS ORIGINAIS ---
    # print(f"\nAdicionando os {len(segmentos_normais_treino)} segmentos normais...") # <-- REMOVIDO
    for chave_normal, df_normal in segmentos_normais_treino.items():
        lista_sinais_treino.append({
            'sinal_final': df_normal['amplitude'].values,
            'tipo_falha_adicionada': 'Normal',
            'rpm': df_normal['rotacao_rpm'].iloc[0],
            'multiplicador_amplitude': 0,
            'fase_adicionada_rad': 0,
            'base_normal': chave_normal
        })

    # --- 6. DATAFRAME INTERMEDIÁRIO ---
    df_sinais_treino = pd.DataFrame(lista_sinais_treino)

    print(f"Geração concluída. Total de {len(df_sinais_treino)} segmentos de treino.") # <-- MODIFICADO
    
    return df_sinais_treino

# %% [markdown]
# ## Extração de atributos

# %%
from scipy.stats import skew

# =============================================================================
# BLOCO 3 (Refatorado): FUNÇÃO DE EXTRAÇÃO DE FEATURES
# =============================================================================

def extrair_features_treino_teste(
    df_sinais_treino, 
    dicionario_teste, 
    TAXA_AMOSTRAL,
    min_freq_pico=50, 
    max_freq_pico=200
):
    """
    Extrai atributos (versão 'limpa', com menos prints).
    """

    # --- 1. FUNÇÕES AUXILIARES INTERNAS ---

    def calcular_tf2_std(sinal): return np.std(sinal)
    def calcular_tf3_rms(sinal): return np.sqrt(np.mean(sinal**2))
    def calcular_tf4_fator_forma(sinal):
        rms = calcular_tf3_rms(sinal)
        media_abs = np.mean(np.abs(sinal))
        return rms / media_abs if media_abs != 0 else 0

    def calcular_features_frequencia(sinal, taxa_amostral):
        N = len(sinal)
        if N == 0: return 0, 0, 0
        espectro = np.abs(np.fft.fft(sinal)[0:N//2])
        freqs = np.fft.fftfreq(N, 1 / taxa_amostral)[:N//2]
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
        espectro = np.abs(np.fft.fft(sinal)[0:N//2])
        freqs = np.fft.fftfreq(N, 1 / taxa_amostral)[:N//2]
        range_mask = (freqs >= min_freq) & (freqs <= max_freq)
        freqs_filtradas = freqs[range_mask]
        espectro_filtrado = espectro[range_mask]
        if len(espectro_filtrado) == 0: return 0 
        indice_pico = np.argmax(espectro_filtrado)
        freq_pico = freqs_filtradas[indice_pico]
        return freq_pico

    def extrair_todas_features(sinal, taxa_amostral, min_f, max_f):
        tf2 = calcular_tf2_std(sinal)
        tf3 = calcular_tf3_rms(sinal)
        tf4 = calcular_tf4_fator_forma(sinal)
        ff2, ff3, ff5 = calcular_features_frequencia(sinal, taxa_amostral)
        ff_pico_range = calcular_freq_pico_range(sinal, taxa_amostral, min_freq=min_f, max_freq=max_f)
        return {
            'TF2_std': tf2, 'TF3_rms': tf3, 'TF4_fator_forma': tf4,
            'FF2_freq_central': ff2, 'FF3_rms_freq': ff3, 'FF5_assimetria_espectral': ff5,
            'FF_pico_50_200Hz': ff_pico_range
        }

    # =============================================================================
    # --- 2. PROCESSAMENTO DO CONJUNTO DE TREINO ---
    # =============================================================================
    print("Extraindo features de TREINO...") # <-- MODIFICADO
    lista_de_features_treino = []
    for linha in df_sinais_treino.itertuples():
        features = extrair_todas_features(
            linha.sinal_final, TAXA_AMOSTRAL, min_freq_pico, max_freq_pico
        )
        features['tipo_falha_adicionada'] = linha.tipo_falha_adicionada
        features['rpm'] = linha.rpm
        features['multiplicador_amplitude'] = linha.multiplicador_amplitude
        features['fase_adicionada_rad'] = linha.fase_adicionada_rad
        features['base_normal'] = linha.base_normal
        lista_de_features_treino.append(features)

    df_treino = pd.DataFrame(lista_de_features_treino)
    # print(f"Extração concluída! {len(df_treino)} amostras...") # <-- REMOVIDO

    # =============================================================================
    # --- 3. PROCESSAMENTO DO CONJUNTO DE TESTE ---
    # =============================================================================
    print("Extraindo features de TESTE...") # <-- MODIFICADO
    lista_de_features_teste = []
    for chave, df_segmento in dicionario_teste.items():
        sinal = df_segmento['amplitude'].values
        features = extrair_todas_features(
            sinal, TAXA_AMOSTRAL, min_freq_pico, max_freq_pico
        )
        features['tipo_falha_adicionada'] = df_segmento['tipo_falha'].iloc[0]
        features['rpm'] = df_segmento['rotacao_rpm'].iloc[0]
        features['arquivo_origem'] = df_segmento['arquivo_origem'].iloc[0]
        lista_de_features_teste.append(features)

    df_teste = pd.DataFrame(lista_de_features_teste)
    print(f"Extração concluída. {len(df_treino)} amostras de treino, {len(df_teste)} amostras de teste.") # <-- MODIFICADO

    # --- 4. EXIBIÇÃO DOS RESULTADOS ---
    # Todos os prints de info() e head() foram removidos
    
    return df_treino, df_teste


