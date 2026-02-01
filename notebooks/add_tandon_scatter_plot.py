import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_tandon_vs_baseline(results_df, target_diameter_label=None):
    """
    Plots SAM vs K for the Tandon (FFT) method, comparing it against the Normal Baseline.
    Generates separate figures for each Fault Type and Diameter combination.
    """
    
    # 1. Determine which diameters to plot
    available_diameters = results_df['diameter_label'].unique()
    
    if target_diameter_label is None:
        target_diameters = sorted(available_diameters)
    elif isinstance(target_diameter_label, str):
        target_diameters = [target_diameter_label]
    else:
        target_diameters = target_diameter_label
        
    print(f"Gerando gráficos para os diâmetros: {target_diameters}")

    plt.rcParams.update({'font.size': 14}) # Base font size

    for current_dia in target_diameters:
        if current_dia not in available_diameters:
            print(f"Aviso: Diâmetro '{current_dia}' não encontrado nos dados. Ignorando.")
            continue
            
        df_fft = results_df[
            (results_df['method'] == 'fft') & 
            (results_df['diameter_label'] == current_dia)
        ].copy()
        
        if df_fft.empty:
            continue

        fault_types = df_fft['fault_code'].unique()
        rpms = sorted(df_fft['rpm'].unique())
        palette = sns.color_palette("viridis", len(rpms))
        rpm_colors = dict(zip(rpms, palette))
        
        print(f"--- Processando Diâmetro {current_dia} ---")

        for fault_code in fault_types:
            plt.figure(figsize=(12, 8), dpi=200)
            
            subset = df_fft[df_fft['fault_code'] == fault_code]
            title_name = subset['fault_name'].iloc[0] if not subset.empty else fault_code
            
            # --- Plot Tandon FFT (Scatter + Line) ---
            sns.lineplot(
                data=subset, 
                x='k', 
                y='sam_mean_deg', 
                hue='rpm', 
                style='rpm',
                markers=True, 
                dashes=False,
                palette=rpm_colors,
                linewidth=3,
                markersize=12,
                legend='full' # Force legend generation to capture handles
            )
            
            # Capture Seaborn handles for the Main Legend
            ax = plt.gca()
            if ax.get_legend():
                curve_handles = ax.get_legend().legend_handles
                curve_labels = [t.get_text() for t in ax.get_legend().get_texts()]
                ax.get_legend().remove()
            else:
                curve_handles, curve_labels = ax.get_legend_handles_labels()

            # --- Add Baseline Lines (Normal vs Real) ---
            baseline_handles = []
            baseline_labels = []

            for rpm in rpms:
                rpm_subset = subset[subset['rpm'] == rpm]
                if rpm_subset.empty:
                    continue
                    
                baseline_val = rpm_subset['sam_real_vs_normal_deg'].iloc[0]
                
                line = plt.axhline(
                    y=baseline_val, 
                    color=rpm_colors[rpm], 
                    linestyle='--', 
                    alpha=0.8,
                    linewidth=2,
                    label=f"{rpm}"
                )
                baseline_handles.append(line)
                baseline_labels.append(f"{rpm}")

            # Titles and Labels
            plt.title(f'Sensibilidade Tandon - {title_name}\n(Diâmetro {current_dia})', fontsize=20, pad=20)
            plt.xlabel('Fator de Escala K (Amplitude do Impulso)', fontsize=16, labelpad=15)
            plt.ylabel('SAM Médio (Graus)', fontsize=16, labelpad=15)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(True, linestyle=':', alpha=0.6)
            
            # Legends
            # 1. Main Legend - RPMs for the curves (Explicit handles)
            main_legend = plt.legend(
                curve_handles, curve_labels, 
                title='RPM', 
                title_fontsize=14, 
                fontsize=12, 
                bbox_to_anchor=(1.01, 1), 
                loc='upper left'
            )
            plt.gca().add_artist(main_legend)

            # 2. Baseline Legend - specific for dashed lines
            plt.legend(
                baseline_handles, 
                baseline_labels,
                title='Baseline',
                title_fontsize=14, 
                fontsize=12,
                bbox_to_anchor=(1.01, 0.6), 
                loc='upper left'
            )

            plt.tight_layout()
            plt.show()


def plot_method_comparison(results_df, df_baseline):
    """
    Plots a Bar Chart comparing the Best Performance (Minimum SAM) of each method:
    - Baseline (Normal)
    - Tandon (FFT)
    - Impulse
    
    Also displays a table with the best configuration (K, RPM) for the Tandon method.
    """
    print("Visualização Final e Detalhamento FFT\n")

    # 1. Concatenate Results DataFrames
    cols_interest = ['rpm', 'diameter_mm', 'diameter_label', 'fault_name', 
                     'method', 'sam_mean_deg', 'k']
    
    # Prepare Synthetic Results (FFT, Impulse)
    df_syn = results_df[cols_interest].copy()
    
    # Prepare Baseline Results
    # Baseline doesn't have 'k', so we fill with 0.0
    df_base = df_baseline[cols_interest].copy()
    if 'k' not in df_base.columns: 
        df_base['k'] = 0.0

    # Rename methods as requested
    method_map = {
        'fft': 'Tandon',
        'natural_baseline': 'Baseline',
        'impulse': 'Impulse'
    }
    
    df_syn['method'] = df_syn['method'].map(lambda x: method_map.get(x, x))
    df_base['method'] = df_base['method'].map(lambda x: method_map.get(x, x))

    df_final = pd.concat([df_syn, df_base], ignore_index=True)
    
    unique_faults = df_final['fault_name'].unique()
    
    # Updated Palette with requested labels (Purple & Light Green from Viridis style)
    hue_order = ['Baseline', 'Tandon', 'Impulse']
    custom_palette = {'Baseline': '#999999', 'Tandon': '#440154', 'Impulse': '#5ec962'}

    for fault in unique_faults:
        # Filter by fault
        df_view = df_final[df_final['fault_name'] == fault].copy()
        df_view = df_view.dropna(subset=['sam_mean_deg'])
        
        if df_view.empty:
            continue

        # Select BEST configuration (Min SAM) for each method/diameter
        # We group by [method, diameter_label] and pick the row with min SAM
        best_indices = df_view.groupby(['method', 'diameter_label'])['sam_mean_deg'].idxmin()
        best_data = df_view.loc[best_indices].sort_values(by=['diameter_mm', 'method'])

        # --- PLOT ---
        plt.figure(figsize=(10, 6), dpi=200)
        
        sns.barplot(
            data=best_data, 
            x='diameter_label', 
            y='sam_mean_deg', 
            hue='method',
            hue_order=[h for h in hue_order if h in best_data['method'].values],
            palette=custom_palette
        )
        plt.title(f'Comparação de Desempenho - {fault}', fontsize=16)
        plt.xlabel('Diâmetro da Falha', fontsize=14)
        plt.ylabel('Menor Ângulo SAM (Graus)', fontsize=14)
        plt.grid(False)
        plt.legend(title='Método', fontsize=12, title_fontsize=13, loc='lower right', framealpha=0.9)
        plt.tight_layout()
        plt.show()

        # --- DETAIL TABLE (Tandon Only) ---
        print(f"--- Melhores Configurações Tandon para {fault} ---")
        
        # Filter only Tandon rows from the best data
        fft_details = best_data[best_data['method'] == 'Tandon'].copy()
        
        if not fft_details.empty:
            # Select and rename columns for clean display
            display_cols = ['diameter_label', 'rpm', 'k', 'sam_mean_deg']
            fft_table = fft_details[display_cols].reset_index(drop=True)
            display(fft_table)
        else:
            print("Nenhum dado Tandon válido encontrado para este defeito.")
        
        print("\n" + "="*80 + "\n")

print("Funções carregadas: 'plot_tandon_vs_baseline', 'plot_tandon_facet_grid', 'plot_tandon_diameter_comparison', 'plot_method_comparison'.")
