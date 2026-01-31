import json
import os

nb_path = r'c:/Users/Cliente/Documents/GitHub/TCC/notebooks/compare_synthetic_methods.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Global Config Cell
# We look for the cell containing 'UPPER_FREQ_LIMIT = 800'
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'UPPER_FREQ_LIMIT = 800' in source:
            new_source = source.replace(
                'Y_MAX_AMPLITUDE  = 0.02    # Ajuste conforme necesário para escala fixa',
                'Y_MAX_AMPLITUDE  = 0.02    # Ajuste conforme necesário para escala fixa\nSAMPLES_TO_ANALYZE = 1024 # Limitando amostras'
            )
            cell['source'] = new_source.splitlines(keepends=True)
            print("Updated Global Config.")
            break

# 2. Update Truncation Logic
# Look for 'duration = n / FS' and insert truncation before it
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'if normal_seg is None:' in source and 'duration = n / FS' in source:
            # We insert the truncation logic before 'n = len(normal_seg)'
            parts = source.split('n = len(normal_seg)')
            if len(parts) == 2:
                truncation_code = '\n# Truncar o sinal\nif len(normal_seg) > SAMPLES_TO_ANALYZE:\n    normal_seg = normal_seg[:SAMPLES_TO_ANALYZE]\n    print(f"Sinal truncado para {SAMPLES_TO_ANALYZE} amostras.")\n\n'
                new_source = parts[0] + truncation_code + 'n = len(normal_seg)' + parts[1]
                cell['source'] = new_source.splitlines(keepends=True)
                print("Updated Truncation Logic.")
                break

# 3. Update plot_comparison function
# We replace the whole cell containing 'def plot_comparison'
new_plot_function = r"""def plot_comparison(fault_name, sig_fft_comb, sig_impulse_comb, characteristic_freq=None, sig_real=None, f_min=LOWER_FREQ_LIMIT, f_max=UPPER_FREQ_LIMIT):
    # Truncar sinais se necessário (garantia extra)
    if 'SAMPLES_TO_ANALYZE' in globals():
        sig_fft_comb = sig_fft_comb[:SAMPLES_TO_ANALYZE]
        sig_impulse_comb = sig_impulse_comb[:SAMPLES_TO_ANALYZE]
        if sig_real is not None:
            sig_real = sig_real[:SAMPLES_TO_ANALYZE]

    # Lógica de escala Y dinâmica
    if "Esfera" in fault_name or "Ball" in fault_name:
        y_max = 0.020
    else:
        y_max = 0.020

    # Função interna para processar FFT com o novo filtro
    def calc_fft_and_process(sig):
        # 1. Filtro Passa-Baixa (NOVO)
        sig_lp = apply_lowpass_filter(sig, FS, cutoff_freq=f_max)
        
        # 2. Antialiasing (Padrão)
        sig_filt = apply_antialiasing_filter(sig_lp, FS)

        # 3. Janela de Hanning
        sig_win = apply_hanning_window(sig_filt)
        
        # 4. FFT
        yf = fft(sig_win)
        xf = fftfreq(len(sig_win), 1/FS)[:len(sig_win)//2]
        mag = 2.0/len(sig_win) * np.abs(yf[0:len(sig_win)//2])
        
        return xf, mag

    # Configuração comum de eixos para todos os subplots
    def setup_axis(ax, title):
        ax.set_title(title, fontsize=13, fontweight='bold') # +5%
        ax.set_ylabel("Magnitude", fontsize=11)             # +5%
        ax.set_xlabel("Frequência (Hz)", fontsize=11)       # +5%
        ax.set_xlim(f_min, f_max)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)            # +5%

    # Plot 1: Método FFT (Figura Separada) - Achatado +5% (5.7x4)
    plt.figure(figsize=(5.7, 4), dpi=200) # Resolução melhor (200)
    x1, y1 = calc_fft_and_process(sig_fft_comb)
    plt.plot(x1, y1, color='black', alpha=0.7, label='Sinal Sintético (FFT)')
    if characteristic_freq:
        plt.axvline(x=characteristic_freq, color='red', linestyle='--', alpha=0.5, label=f'Falha: {characteristic_freq:.1f}Hz')
    setup_axis(plt.gca(), f"{fault_name} - Método de Tandon (FFT)")
    plt.tight_layout()
    plt.show()

    # Plot 2: Método Impulso (Figura Separada) - Achatado +5% (5.7x4)
    plt.figure(figsize=(5.7, 4), dpi=200)
    x2, y2 = calc_fft_and_process(sig_impulse_comb)
    plt.plot(x2, y2, color='black', alpha=0.7, label='Sinal Sintético (Impulso)')
    if characteristic_freq:
        plt.axvline(x=characteristic_freq, color='red', linestyle='--', alpha=0.5, label=f'Falha: {characteristic_freq:.1f}Hz')
    setup_axis(plt.gca(), f"{fault_name} - Método de Impulso")
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Sinal Real (Figura Separada) - Achatado +5% (5.7x4)
    if sig_real is not None:
        plt.figure(figsize=(5.7, 4), dpi=200)
        x3, y3 = calc_fft_and_process(sig_real)
        plt.plot(x3, y3, color='blue', alpha=0.7, label='Sinal Real')
        if characteristic_freq:
            plt.axvline(x=characteristic_freq, color='red', linestyle='--', alpha=0.5, label=f'Falha: {characteristic_freq:.1f}Hz')
        setup_axis(plt.gca(), f"{fault_name} - Sinal Real")
        plt.tight_layout()
        plt.show()
"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'def plot_comparison' in source:
            cell['source'] = new_plot_function.splitlines(keepends=True)
            print("Updated plot_comparison function.")
            break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
