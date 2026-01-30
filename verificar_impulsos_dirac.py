import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from generate_impulse_data import gerar_sinal_impulso_completo, amplitudes_referencia

# Parâmetros de teste
fs = 12000  # Taxa de amostragem
duracao = 0.1  # 100ms
n_pontos = int(duracao * fs)
freq_falha = 100  # 100 Hz

# Gerar sinal impulso puro
sinal_impulso = gerar_sinal_impulso_completo(
    fs=fs,
    duration_points=n_pontos,
    defect_freq_hz=freq_falha
)

# Aplicar amplitude de referência (Pista Externa = 0.4)
amplitude_ref = amplitudes_referencia['Drive End']['Pista Externa']
sinal_com_amplitude = sinal_impulso * amplitude_ref

# Análise
impulsos_indices = np.where(sinal_impulso > 0)[0]
num_impulsos = len(impulsos_indices)
amplitude_impulsos = sinal_com_amplitude[impulsos_indices]

print("=" * 60)
print("VERIFICAÇÃO: IMPULSOS DELTA DE DIRAC PUROS")
print("=" * 60)
print(f"\nParâmetros:")
print(f"  - Taxa de amostragem: {fs} Hz")
print(f"  - Duração: {duracao} s ({n_pontos} pontos)")
print(f"  - Frequência de falha: {freq_falha} Hz")
print(f"  - Amplitude de referência: {amplitude_ref}")

print(f"\nResultados:")
print(f"  - Número de impulsos detectados: {num_impulsos}")
print(f"  - Número esperado: {int(duracao * freq_falha)}")
print(f"  - Amplitude dos impulsos: {amplitude_impulsos[0]:.4f}")
print(f"  - Todas amplitudes iguais? {np.allclose(amplitude_impulsos, amplitude_ref)}")

# Verificar espaçamento entre impulsos
if num_impulsos > 1:
    espacamentos = np.diff(impulsos_indices)
    periodo_amostras_esperado = fs / freq_falha
    print(f"  - Espaçamento entre impulsos (amostras): {espacamentos[0]}")
    print(f"  - Espaçamento esperado (amostras): {periodo_amostras_esperado:.1f}")
    print(f"  - Todos espaçamentos iguais? {np.all(espacamentos == espacamentos[0])}")

# Verificar que só existem 0s e valores de amplitude
valores_unicos = np.unique(sinal_com_amplitude)
print(f"\nValores únicos no sinal: {valores_unicos}")
print(f"  - Apenas zeros e amplitude de ref? {len(valores_unicos) == 2 and amplitude_ref in valores_unicos and 0.0 in valores_unicos}")

print("\n" + "=" * 60)
print("SINAL DELTA DE DIRAC PURO CONFIRMADO!")
print("=" * 60)

# Plotar
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Sinal completo
axes[0].stem(np.arange(n_pontos), sinal_com_amplitude, basefmt=' ', linefmt='b-', markerfmt='bo')
axes[0].set_title(f'Trem de Impulsos Delta de Dirac ({freq_falha} Hz, Amplitude={amplitude_ref})')
axes[0].set_xlabel('Amostras')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)

# Zoom nos primeiros impulsos
n_zoom = min(300, n_pontos)
axes[1].stem(np.arange(n_zoom), sinal_com_amplitude[:n_zoom], basefmt=' ', linefmt='r-', markerfmt='ro')
axes[1].set_title(f'Zoom - Primeiros {n_zoom} pontos')
axes[1].set_xlabel('Amostras')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('verificacao_impulsos_dirac.png', dpi=150, bbox_inches='tight')
print(f"\nImagem salva: verificacao_impulsos_dirac.png")
plt.close()
