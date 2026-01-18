import os
import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to the data directory (robust handling for running from src or root)
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
caminho_raiz = os.path.join(base_path, 'data')

# Mapping dictionaries (extracted from clean-and-transformation.ipynb)
mapa_tipo_falha = {'IR': 'Pista Interna', 'B': 'Esfera', 'OR': 'Pista Externa', 'Normal': 'Normal'}
mapa_diametro_falha = {'7': '0.007"', '14': '0.014"', '21': '0.021"'}

# --- PARÂMETROS DE SEGMENTAÇÃO ---
tamanho_segmento = 4096
sobreposicao_percentual = 0.3
passo = int(tamanho_segmento * (1 - sobreposicao_percentual))

# =============================================================================
# PROCESSAMENTO
# =============================================================================

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
    # Range arguments must be integers
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

            # Robust fault type extraction
            tipo_falha_raw = 'Normal'
            diametro_raw = 'N/A'
            
            if not is_normal:
                if len(partes) > 1:
                    tipo_falha_raw = partes[1].split('@')[0]
                if len(partes) > 2:
                    diametro_raw = partes[2]

            metadados = {
                'nome_arquivo': nome_arquivo,
                'rpm': int(rpm_str) if rpm_str.isdigit() else 0,
                'tipo_falha': 'Normal' if is_normal else mapa_tipo_falha.get(tipo_falha_raw, 'Desconhecido'),
                'diametro_falha': 'N/A' if is_normal else mapa_diametro_falha.get(diametro_raw, 'Desconhecido')
            }

            try:
                dados_npz = np.load(caminho_completo)
                sensor_cod = 'DE' # Foco apenas no Drive End, como no seu código original

                # Check for DE key variants if necessary, but starting with exact match
                target_key = None
                for k in dados_npz.files:
                    if 'DE' in k:
                        target_key = k
                        break
                
                # If specific 'DE' key not found, maybe valid if it's the only one or 'Normal' specific?
                # The user snippet used: if sensor_cod in dados_npz.files:
                # But real files might have specific keys like '1730_DE_time' etc.
                # Let's stick to user's logic but make it slightly robust if 'DE' is a substring
                
                keys_found = [k for k in dados_npz.files if 'DE' in k]
                
                if keys_found:
                    # Use the first matching DE key
                    sinal_completo = dados_npz[keys_found[0]].ravel()

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
                        # Remove digits from the last part if it exists (e.g. DE12 -> DE)
                        if partes_chave:
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
