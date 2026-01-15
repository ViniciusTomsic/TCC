import numpy as np
import pandas as pd

def calculate_tandon_coefficients(fault_diameter_mm, rpm, fault_type, K=1.0):
    """
    Calculates Fourier coefficients (Po, Pr, Fo, Fs) for bearing analysis 
    based on the Tandon and Choudhury (1997) model.

    Parameters:
    -----------
    fault_diameter_mm : float
        Width of the defect in millimeters (b).
    rpm : float
        Shaft rotational speed in Revolutions Per Minute.
    fault_type : str
        Type of fault: 'inner', 'outer', or 'ball'.
    K : float, optional
        Pulse height coefficient (default=1.0 for normalized analysis).

    Returns:
    --------
    dict
        A dictionary containing:
        - 'Po': Load distribution mean coefficient (float)
        - 'Pr': Function to calculate r-th harmonic of load (callable)
        - 'Fo': Pulse mean coefficient (float)
        - 'Fs': Function to calculate s-th harmonic of pulse (callable)
        - 'frequencies': Dictionary of characteristic frequencies (Hz)
        - 'debug_info': Intermediate values (pulse width, relative velocity, etc.)
    """

    # ==========================================
    # 1. CONSTANTS & BEARING GEOMETRY (6205-2 RS)
    # ==========================================
    # Dimensions converted to meters for calculation consistency
    D_ext = 52.0 * 1e-3     # Outer Diameter
    D_int = 25.0 * 1e-3     # Inner Diameter
    W = 15.0 * 1e-3         # Width
    d = 7.94 * 1e-3         # Ball diameter (d) [cite: 1054]
    D = 39.05 * 1e-3        # Pitch diameter (D) [cite: 1054]
    Z = 9                   # Number of balls [cite: 1054]
    alpha_deg = 0.0         # Contact angle (assumed 0 for zero clearance)
    alpha = np.radians(alpha_deg)
    
    # Load parameters
    radial_load_kg = 100
    radial_load_N = radial_load_kg * 9.80665 # Convert kg to Newtons
    
    # Pulse Height (K) - Now configurable via parameter (default 1.0) 

    # ==========================================
    # 2. KINEMATIC CALCULATIONS
    # ==========================================
    # Shaft angular velocity (rad/s)
    w_s = (2 * np.pi * rpm) / 60.0  
    f_s = rpm / 60.0 # Shaft frequency Hz

    # Cage angular velocity (rad/s) - Eq in Appendix I [cite: 1859]
    # w_c = (w_s / 2) * (1 - (d/D) * cos(alpha))
    w_c = (w_s / 2.0) * (1.0 - (d / D) * np.cos(alpha))

    # Ball Spin angular velocity (rad/s) - Eq in Appendix I [cite: 1860]
    w_b = (D * w_s / (2.0 * d)) * (1.0 - (d**2 / D**2) * (np.cos(alpha)**2))

    # Determine Characteristic Defect Frequency (rad/s) based on fault type [cite: 1861, 1863, 1864]
    if fault_type.lower() == 'inner':
        # Inner race defect frequency: Z(w_s - w_c)
        w_defect = Z * (w_s - w_c) 
    elif fault_type.lower() == 'outer':
        # Outer race defect frequency: Z * w_c
        w_defect = Z * w_c
    elif fault_type.lower() == 'ball':
        # Ball defect frequency: 2 * w_b
        w_defect = 2.0 * w_b
    else:
        raise ValueError("fault_type must be 'inner', 'outer', or 'ball'")

    # Period of the pulse (T)
    # Note: For inner/outer race, T is time between ball passes. 
    # For ball fault, T is time between impacts on races.
    T = 2 * np.pi / w_defect

    # ==========================================
    # 3. PULSE WIDTH & DUTY CYCLE
    # ==========================================
    # Relative Velocity (v_r) - Tandon Eq. 11 [cite: 1462]
    # v_r = (D * w_s / 4) * (1 - (d^2 / D^2) * cos^2(alpha))
    # Note: This specific formula is unique to Tandon's relative velocity calculation.
    v_r = (D * w_s / 4.0) * (1.0 - (d**2 / D**2) * (np.cos(alpha)**2))
    
    # Defect width (b) in meters
    b = fault_diameter_mm * 1e-3

    # Pulse Width (Delta T) - Tandon Eq. 12 [cite: 1463]
    delta_T = b / v_r

    # Duty Cycle (m) - Appendix II [cite: 1874]
    m = delta_T / T

    # Safety check: duty cycle cannot exceed 1
    if m > 1.0:
        print(f"Warning: Defect size is too large for the RPM. Duty cycle {m:.2f} clipped to 1.0")
        m = 1.0

    # ==========================================
    # 4. LOAD DISTRIBUTION COEFFICIENTS (Po, Pr)
    # ==========================================
    # Max Load (P_max) - Stribeck Equation (Harris 1966 via Tandon Eq 3.1) [cite: 1208]
    # P_max = (4.37 * F_r) / (Z * cos(alpha))
    P_max = (4.37 * radial_load_N) / (Z * np.cos(alpha))

    # Load Zone Parameters for Zero Clearance
    # epsilon = 0.5 
    # phi_1 = pi/2 (Load zone extent)
    epsilon = 0.5
    phi_1 = np.pi / 2.0
    
    # Ball bearing load-deflection exponent (n)
    n_exp = 1.5 

    # Coefficients A0, A1, A2, A3 - Tandon Eq. 8b [cite: 1443]
    # Note: These are derived from binomial expansion of load distribution
    A0 = 1.0 - (n_exp / (2*epsilon)) + (n_exp*(n_exp-1) / (8*epsilon**2)) * 1.5 - (n_exp*(n_exp-1)*(n_exp-2) / (48*epsilon**3)) * 2.5
    A1 = (n_exp / (2*epsilon)) - (n_exp*(n_exp-1) / (8*epsilon**2)) * 2.0 + (n_exp*(n_exp-1)*(n_exp-2) / (48*epsilon**3)) * 3.75
    A2 = (n_exp*(n_exp-1) / (8*epsilon**2)) * 0.5 - (n_exp*(n_exp-1)*(n_exp-2) / (48*epsilon**3)) * 1.5
    A3 = (n_exp*(n_exp-1)*(n_exp-2) / (48*epsilon**3)) * 0.25
    
    # Store A coefficients in a list for easier summation in Pr
    A_coeffs = [A0, A1, A2, A3]

    # Calculate Po - Tandon Eq. 9 [cite: 1446]
    # Po = (P_max / pi) * [A0*phi1 + A1*sin(phi1) + (A2/2)*sin(2*phi1) + (A3/3)*sin(3*phi1)]
    term_Po = A0 * phi_1 + A1 * np.sin(phi_1) + (A2/2.0) * np.sin(2.0*phi_1) + (A3/3.0) * np.sin(3.0*phi_1)
    Po = (P_max / np.pi) * term_Po

    # Define function for Pr (Load Harmonics) - Tandon Eq. 10 [cite: 1446]
    def get_Pr(r):
        """Calculates the r-th Fourier coefficient for Load P."""
        if r == 0:
            return Po
        
        # Term 1: 2*A0/r * sin(r*phi1)
        term1 = (2.0 * A0 / r) * np.sin(r * phi_1)
        
        # Term 2: Summation for l=1 to 3
        term2 = 0.0
        for l in range(1, 4): # l = 1, 2, 3
            Al = A_coeffs[l]
            
            # Handling singularity if r == l (though r is usually integer > 0)
            if r == l:
                # Limit sin(x)/x as x->0 is 1. But here denominator is (r-l).
                # Term becomes phi1 * cos(0) = phi1
                val_minus = phi_1 
            else:
                val_minus = np.sin((r - l) * phi_1) / (r - l)
                
            val_plus = np.sin((r + l) * phi_1) / (r + l)
            
            term2 += Al * (val_plus + val_minus)
            
        return (P_max / np.pi) * (term1 + term2)

    # ==========================================
    # 5. PULSE SHAPE COEFFICIENTS (Fo, Fs)
    # ==========================================
    # Rectangular Pulse Model - Appendix II [cite: 1872, 1873]
    
    # Fo = K * m [cite: 1873]
    Fo = K * m

    # Define function for Fs (Pulse Harmonics) - Appendix II [cite: 1875]
    def get_Fs(s):
        """Calculates the s-th Fourier coefficient for Pulse F (Rectangular)."""
        if s == 0:
            return Fo
        
        # Fs = (2*K / (pi*s)) * sin(pi*s*m)
        return (2.0 * K / (np.pi * s)) * np.sin(np.pi * s * m)

    # ==========================================
    # 6. PACKAGING RESULTS
    # ==========================================
    return {
        "Po": Po,
        "Pr": get_Pr,  # Callable function: Pr(r)
        "Fo": Fo,
        "Fs": get_Fs,  # Callable function: Fs(s)
        "frequencies": {
            "shaft_freq_hz": w_s / (2*np.pi),
            "cage_freq_hz": w_c / (2*np.pi),
            "defect_freq_hz": w_defect / (2*np.pi)
        },
        "debug_info": {
            "v_r_m_s": v_r,
            "delta_T_sec": delta_T,
            "period_T_sec": T,
            "duty_cycle_m": m,
            "P_max_N": P_max,
            "load_coeffs_A": A_coeffs
        }
    }


def get_bearing_natural_frequencies():
    """
    Calculates natural frequencies and mass for the Outer and Inner races
    of a 6205-2 RS bearing using Tandon's Equations 1 & 15.
    
    Returns:
        df_modes (pd.DataFrame): Dataframe containing frequencies, mass, 
                                 and geometry for each mode.
    """
    # ---------------------------------------------------------
    # 1. Fixed Parameters (User Provided)
    # ---------------------------------------------------------
    D_ext = 52.0 * 1e-3     # Outer Diameter (m)
    D_int = 25.0 * 1e-3     # Inner Diameter (m)
    W = 15.0 * 1e-3         # Width (m)
    d = 7.94 * 1e-3         # Ball diameter (m)
    D = 39.05 * 1e-3        # Pitch diameter (m)
    Z = 9                   # Number of balls
    
    # Material Properties (Standard Bearing Steel)
    rho_vol = 7850          # Density (kg/m^3)
    E = 2.07e11             # Young's Modulus (Pa)

    # ---------------------------------------------------------
    # 2. Geometry Approximation (Rectangular Cross-Section)
    # ---------------------------------------------------------
    # We estimate the effective thickness (h) for the "equivalent rectangular ring"
    # assumption used by Tandon.
    
    # --- Outer Race Geometry ---
    # Max thickness (at shoulder) approx: (D_ext - Pitch) / 2
    h_out_max = (D_ext - D) / 2.0
    # Min thickness (at groove bottom) approx: (D_ext - (Pitch + Ball)) / 2
    h_out_min = (D_ext - (D + d)) / 2.0
    # Effective thickness (average)
    h_outer = (h_out_max + h_out_min) / 2.0
    # Neutral axis radius (a)
    a_outer = (D_ext / 2.0) - (h_outer / 2.0)

    # --- Inner Race Geometry ---
    # Max thickness approx: (Pitch - D_int) / 2
    h_in_max = (D - D_int) / 2.0
    # Min thickness approx: ((Pitch - Ball) - D_int) / 2
    h_in_min = ((D - d) - D_int) / 2.0
    # Effective thickness
    h_inner = (h_in_max + h_in_min) / 2.0
    # Neutral axis radius (a)
    a_inner = (D_int / 2.0) + (h_inner / 2.0)

    # ---------------------------------------------------------
    # 3. Calculation Function
    # ---------------------------------------------------------
    def calc_race_modes(race_name, h, a, n_modes=1000):
        results = []
        
        # Section Properties
        I = (W * h**3) / 12.0       # Area Moment of Inertia (m^4)
        A_cs = W * h                # Cross-sectional Area (m^2)
        rho_lin = rho_vol * A_cs    # Mass per unit length (kg/m)
        
        # Generalized Mass (M_i) - Equation 15 (Tandon)
        # "Equals the actual mass of the ring" and is constant for all modes.
        M_i = 2 * np.pi * rho_lin * a 
        
        for i in range(2, n_modes + 1): # Modes i = 2, 3, 4, ...
            
            # Natural Frequency - Equation 1 (Tandon)
            # w_i = (i(i^2-1) / sqrt(1+i^2)) * sqrt(EI / rho a^4)
            term_mode = (i * (i**2 - 1)) / np.sqrt(1 + i**2)
            term_phys = np.sqrt((E * I) / (rho_lin * a**4))
            
            w_n_rad = term_mode * term_phys
            f_n_hz = w_n_rad / (2 * np.pi)
            
            results.append({
                'Race': race_name,
                'Mode_i': i,
                'Freq_Hz': round(f_n_hz, 2),
                'Freq_rad_s': round(w_n_rad, 2),
                'Mass_kg': round(M_i, 5),
                'I_m4': I,
                'radius_a_m': a
            })
        return results

    # ---------------------------------------------------------
    # 4. Execute & Store
    # ---------------------------------------------------------
    data_outer = calc_race_modes("Outer", h_outer, a_outer)
    data_inner = calc_race_modes("Inner", h_inner, a_inner)
    
    df = pd.DataFrame(data_outer + data_inner)
    return df


# =============================================================================
# NOVA FUNÇÃO UNIFICADA (SOLICITAÇÃO DO USUÁRIO)
# =============================================================================

def calcular_espectro_inner_completo(
    fault_diameter_mm, 
    rpm, 
    max_harmonics=5,     # Quantos picos principais (j)
    num_sidebands=3,     # Quantos sidebands (r) ao redor de cada j
    max_s_iter=50,       # Limite da somatória interna do pulso
    K=1.0                # Pulse height coefficient
):
    """
    Calcula o espectro COMPLETO (Picos Principais + Sidebands) para falha de Pista Interna.
    Mescla as lógicas de 'calcular_amplitude_inner_eq28_avancada' e 'calcular_sidebands_inner_eq_avancada'.
    
    Retorna:
        pd.DataFrame contendo todas as frequências (Principais e Sidebands) calculadas,
        ordenadas por frequência.
    """
    
    # ---------------------------------------------------------
    # 1. Obter Coeficientes de Tandon e Frequências
    # ---------------------------------------------------------
    tandon_data = calculate_tandon_coefficients(fault_diameter_mm, rpm, fault_type='inner', K=K)
    
    Po = tandon_data['Po']
    Fo = tandon_data['Fo']
    
    # Funções callable
    Pr_func = tandon_data['Pr']
    Fs_func = tandon_data['Fs']
    
    # Frequências
    freqs = tandon_data['frequencies']
    bpfi_hz = freqs['defect_freq_hz']
    f_shaft_hz = freqs['shaft_freq_hz']
    
    # ---------------------------------------------------------
    # 2. Obter Modos Naturais e Criar Lookup de Receptância
    # ---------------------------------------------------------
    df_nat_freq = get_bearing_natural_frequencies() 
    df_inner = df_nat_freq[df_nat_freq['Race'] == 'Inner'].reset_index(drop=True)
    
    dict_modes = {}
    for _, row in df_inner.iterrows():
        idx = int(row['Mode_i'])
        if row['Freq_rad_s'] > 0:
            dict_modes[idx] = 1.0 / (row['Mass_kg'] * (row['Freq_rad_s']**2))
        else:
            dict_modes[idx] = 0.0
            
    def get_receptance(k):
        # Indice físico é absoluto
        k_abs = abs(int(k))
        return dict_modes.get(k_abs, 0.0)

    Z = 9  # Número de esferas
    results = []
    
    # ---------------------------------------------------------
    # 3. Loop: Unificado (Para cada Harmônico Principal j)
    # ---------------------------------------------------------
    
    for j in range(1, max_harmonics + 1):
        
        # --- PARTE A: CALCULAR PICO PRINCIPAL (Sideband_r = 0) ---
        # Lógica da Eq. 28 Avancada
        
        f_main_hz = j * bpfi_hz
        w_main_rad = 2 * np.pi * f_main_hz
        idx_base_main = Z * j
        
        # Termo 1 (DC)
        recept_main = get_receptance(idx_base_main)
        term_1_main = (Z * Po * Fo) * recept_main
        
        # Termo 2 (AC - Soma sobre s)
        term_2_sum_main = 0.0
        for s in range(1, max_s_iter + 1):
            Fs_val = Fs_func(s)
            
            # (Zj +/- s)
            recept_plus = get_receptance(idx_base_main + s)
            recept_minus = get_receptance(idx_base_main - s)
            
            # Contribuição
            term_s = (Z * Po * Fs_val / 2.0) * (recept_plus + recept_minus)
            term_2_sum_main += term_s
            
        Y_main_total = term_1_main + term_2_sum_main
        Acc_main = abs(Y_main_total * (w_main_rad**2))
        
        results.append({
            'Harmonic_j': j,
            'Type': 'Main',
            'Sideband_r': 0,
            'Frequency_Hz': f_main_hz,
            'Amplitude_Accel_m_s2': Acc_main
        })
        
        # --- PARTE B: CALCULAR SIDEBANDS (Upper/Lower) ---
        # Lógica da Eq. Sidebands Avancada
        
        for r_abs in range(1, num_sidebands + 1):
            # Pr assume simetria
            Pr_val = Pr_func(r_abs)
            
            for sideband_sign in [-1, 1]:
                r = r_abs * sideband_sign
                sb_type = "Sideband Upper" if r > 0 else "Sideband Lower"
                
                f_sb_hz = (j * bpfi_hz) + (r * f_shaft_hz)
                
                # Ignorar frequências negativas
                if f_sb_hz <= 0:
                    continue
                
                w_sb_rad = 2 * np.pi * f_sb_hz
                
                # Índice base para Sideband: (Z*j - r)
                idx_base_sb = (Z * j) - r
                
                # Termo 1 (DC)
                recept_sb = get_receptance(idx_base_sb)
                Y_sb_term1 = (Z * Pr_val * Fo / 2.0) * recept_sb
                
                # Termo 2 (AC - Soma sobre s)
                Y_sb_term2_sum = 0.0
                for s in range(1, max_s_iter + 1):
                    Fs_val = Fs_func(s)
                    
                    # (Zj - r) +/- s
                    recept_m = get_receptance(idx_base_sb - s)
                    recept_p = get_receptance(idx_base_sb + s)
                    
                    term_s_sb = (Z * Pr_val * Fs_val / 4.0) * (recept_m + recept_p)
                    Y_sb_term2_sum += term_s_sb
                    
                Y_sb_total = Y_sb_term1 + Y_sb_term2_sum
                Acc_sb = abs(Y_sb_total * (w_sb_rad**2))
                
                results.append({
                    'Harmonic_j': j,
                    'Type': sb_type,
                    'Sideband_r': r_abs,
                    'Frequency_Hz': f_sb_hz,
                    'Amplitude_Accel_m_s2': Acc_sb
                })
                
    # ---------------------------------------------------------
    # 4. Finalização
    # ---------------------------------------------------------
    df = pd.DataFrame(results)
    # Ordenar por frequência para visualização clara do espectro
    df = df.sort_values(by='Frequency_Hz').reset_index(drop=True)
    return df


# =============================================================================
# FUNÇÃO OUTER RACE (MOVIDA DO NOTEBOOK)
# =============================================================================

def calcular_espectro_outer_race(fault_diameter_mm, rpm, max_harmonics=10, K=1.0):
    """
    Calcula o espectro de aceleração (teórico) para falha na Pista Externa (Outer Race)
    localizada no centro da zona de carga, baseando-se na Eq. 20 de Tandon.
    
    Dependências:
    - calculate_tandon_coefficients
    - get_bearing_natural_frequencies
    """
    
    # 1. Obter Coeficientes de Tandon (Po, Fs, Pmax, etc.)
    # Chamamos a função passando 'outer' para configurar as frequências corretamente
    tandon_data = calculate_tandon_coefficients(fault_diameter_mm, rpm, fault_type='outer', K=K)
    
    # Extração de parâmetros críticos
    P_max = tandon_data['debug_info']['P_max_N']  # Carga máxima (centro da zona)
    func_Fs = tandon_data['Fs']                    # Função callable para harmônicos do pulso
    fc_Hz = tandon_data['frequencies']['cage_freq_hz'] # Frequência da gaiola
    
    # 2. Obter Propriedades Modais (Frequências Naturais e Massas)
    df_nat_freq = get_bearing_natural_frequencies()
    # Filtramos apenas para o anel externo (Outer)
    df_outer = df_nat_freq[df_nat_freq['Race'] == 'Outer']
    
    # Constante Z (Número de esferas) - Hardcoded em 9
    Z = 9 
    
    # Listas para armazenar o espectro
    freqs_hz = []
    ampls_accel = []
    
    # 3. Loop pelos Harmônicos da BPFO (j = 1, 2, ..., max_harmonics)
    # A falha de pista externa gera picos em Z * fc, 2Z * fc, etc.
    for j in range(1, max_harmonics + 1):
        
        # --- Passo A: Frequência do Harmônico ---
        # Frequência de interesse: j-ésimo harmônico da BPFO
        f_harmonic_Hz = j * Z * fc_Hz
        w_harmonic_rad = 2 * np.pi * f_harmonic_Hz
        
        # O índice 's' para buscar o coeficiente de Fourier do pulso é Z*j
        s = Z * j
        
        # --- Passo B: Força de Excitação ---
        # Coeficiente do pulso para este harmônico
        F_zj = func_Fs(s)
        
        # Força Base = P_max * Z * F_zj
        # P_max é usado pois a falha está no centro da zona de carga (xi = 0)
        force_component = P_max * Z * F_zj
        
        # --- Passo C: Receptância (Somatório dos Modos) ---
        # Eq 20: Soma [ Xi(xi) / (Mi * wi^2) ]
        # Para xi=0, Xi(xi) = 1.
        sum_receptance = 0.0
        
        for _, row in df_outer.iterrows():
            wi = row['Freq_rad_s']
            Mi = row['Mass_kg']
            
            # Adiciona a contribuição deste modo (1 / k_dinamico)
            if wi > 0:
                sum_receptance += 1.0 / (Mi * (wi**2))
        
        # --- Passo D: Deslocamento (Y) ---
        Y_displacement_m = force_component * sum_receptance
        
        # --- Passo E: Aceleração (A) ---
        # A = Y * w^2 (m/s²)
        accel_ms2 = Y_displacement_m * (w_harmonic_rad**2)
        
        freqs_hz.append(f_harmonic_Hz)
        ampls_accel.append(accel_ms2)
        
    # Retorna DataFrame
    return pd.DataFrame({
        'Harmonic_Order': range(1, max_harmonics + 1),
        'Frequency_Hz': freqs_hz,
        'Amplitude_m_s2': ampls_accel
    })


# =============================================================================
# FUNÇÃO BALL BEARING (NOVA - SEGUINDO EQUAÇÕES DO USUÁRIO)
# =============================================================================

def calcular_espectro_ball_completo(
    fault_diameter_mm, 
    rpm, 
    max_harmonics=5,     # Quantos picos principais (j)
    num_sidebands=3,     # Quantos sidebands (r) ao redor de cada j
    max_s_iter=50,       # Limite da somatória interna do pulso
    K=1.0                # Pulse height coefficient
):
    """
    Calcula o espectro COMPLETO (Picos Principais + Sidebands) para falha de Esfera (Ball).
    
    Main peaks at: 2*j*ω_b
    Sidebands at: 2*j*ω_b ± r*ω_c
    
    A diferença para inner race é que ball defects excitam AMBOS os anéis (inner e outer),
    então temos termos M_{2j} (inner) e M'_{2j} (outer) nas equações.
    
    Retorna:
        pd.DataFrame contendo todas as frequências (Principais e Sidebands) calculadas,
        ordenadas por frequência.
    """
    
    # ---------------------------------------------------------
    # 1. Obter Coeficientes de Tandon e Frequências
    # ---------------------------------------------------------
    tandon_data = calculate_tandon_coefficients(fault_diameter_mm, rpm, fault_type='ball', K=K)
    
    Po = tandon_data['Po']
    Fo = tandon_data['Fo']
    
    # Funções callable
    Pr_func = tandon_data['Pr']
    Fs_func = tandon_data['Fs']
    
    # Frequências
    freqs = tandon_data['frequencies']
    ball_spin_hz = freqs['defect_freq_hz'] / 2  # ω_b = (defect_freq) / 2, pois defect_freq = 2*ω_b
    cage_freq_hz = freqs['cage_freq_hz']
    
    # ---------------------------------------------------------
    # 2. Obter Modos Naturais (INNER e OUTER) e Criar Lookups
    # ---------------------------------------------------------
    df_nat_freq = get_bearing_natural_frequencies() 
    df_inner = df_nat_freq[df_nat_freq['Race'] == 'Inner'].reset_index(drop=True)
    df_outer = df_nat_freq[df_nat_freq['Race'] == 'Outer'].reset_index(drop=True)
    
    # Dicionário para Inner (M_{idx})
    dict_modes_inner = {}
    for _, row in df_inner.iterrows():
        idx = int(row['Mode_i'])
        if row['Freq_rad_s'] > 0:
            dict_modes_inner[idx] = 1.0 / (row['Mass_kg'] * (row['Freq_rad_s']**2))
        else:
            dict_modes_inner[idx] = 0.0
    
    # Dicionário para Outer (M'_{idx})
    dict_modes_outer = {}
    for _, row in df_outer.iterrows():
        idx = int(row['Mode_i'])
        if row['Freq_rad_s'] > 0:
            dict_modes_outer[idx] = 1.0 / (row['Mass_kg'] * (row['Freq_rad_s']**2))
        else:
            dict_modes_outer[idx] = 0.0
            
    def get_receptance_inner(k):
        k_abs = abs(int(k))
        return dict_modes_inner.get(k_abs, 0.0)
    
    def get_receptance_outer(k):
        k_abs = abs(int(k))
        return dict_modes_outer.get(k_abs, 0.0)

    results = []
    
    # ---------------------------------------------------------
    # 3. Loop: Unificado (Para cada Harmônico Principal j)
    # ---------------------------------------------------------
    
    for j in range(1, max_harmonics + 1):
        
        # --- PARTE A: CALCULAR PICO PRINCIPAL (Sideband_r = 0) ---
        # Frequência: 2*j*ω_b
        f_main_hz = 2 * j * ball_spin_hz
        w_main_rad = 2 * np.pi * f_main_hz
        
        # Índice base: 2j
        idx_base_main = 2 * j
        
        # Termo 1 (DC): P_o * F_o * [1/(M_{2j}*ω²) + 1/(M'_{2j}*ω'²)]
        recept_inner = get_receptance_inner(idx_base_main)
        recept_outer = get_receptance_outer(idx_base_main)
        term_1_main = Po * Fo * (recept_inner + recept_outer)
        
        # Termo 2 (AC - Soma sobre s): Sum_s (P_o/2) * F_s * [inner + outer]
        term_2_sum_main = 0.0
        for s in range(1, max_s_iter + 1):
            Fs_val = Fs_func(s)
            
            # (2j +/- s) para inner
            recept_inner_plus = get_receptance_inner(idx_base_main + s)
            recept_inner_minus = get_receptance_inner(idx_base_main - s)
            
            # (2j +/- s) para outer
            recept_outer_plus = get_receptance_outer(idx_base_main + s)
            recept_outer_minus = get_receptance_outer(idx_base_main - s)
            
            # Soma contribuições
            term_s = (Po / 2.0) * Fs_val * (
                recept_inner_plus + recept_inner_minus + 
                recept_outer_plus + recept_outer_minus
            )
            term_2_sum_main += term_s
            
        Y_main_total = term_1_main + term_2_sum_main
        Acc_main = abs(Y_main_total * (w_main_rad**2))
        
        results.append({
            'Harmonic_j': j,
            'Type': 'Main',
            'Sideband_r': 0,
            'Frequency_Hz': f_main_hz,
            'Amplitude_Accel_m_s2': Acc_main
        })
        
        # --- PARTE B: CALCULAR SIDEBANDS (Upper/Lower) ---
        # Frequências: 2*j*ω_b ± r*ω_c
        
        for r_abs in range(1, num_sidebands + 1):
            # Pr assume simetria
            Pr_val = Pr_func(r_abs)
            
            for sideband_sign in [-1, 1]:
                r = r_abs * sideband_sign
                sb_type = "Sideband Upper" if r > 0 else "Sideband Lower"
                
                f_sb_hz = (2 * j * ball_spin_hz) + (r * cage_freq_hz)
                
                # Ignorar frequências negativas
                if f_sb_hz <= 0:
                    continue
                
                w_sb_rad = 2 * np.pi * f_sb_hz
                
                # Índice base para Sideband: 2j (não muda com r, diferente do inner)
                idx_base_sb = 2 * j
                
                # Termo 1 (DC): (P_r/2) * F_o * [inner + outer]
                recept_inner_sb = get_receptance_inner(idx_base_sb)
                recept_outer_sb = get_receptance_outer(idx_base_sb)
                Y_sb_term1 = (Pr_val / 2.0) * Fo * (recept_inner_sb + recept_outer_sb)
                
                # Termo 2 (AC - Soma sobre s): Sum_s (P_r/2) * F_s * [inner + outer]
                Y_sb_term2_sum = 0.0
                for s in range(1, max_s_iter + 1):
                    Fs_val = Fs_func(s)
                    
                    # (2j +/- s) para inner
                    recept_inner_m = get_receptance_inner(idx_base_sb - s)
                    recept_inner_p = get_receptance_inner(idx_base_sb + s)
                    
                    # (2j +/- s) para outer
                    recept_outer_m = get_receptance_outer(idx_base_sb - s)
                    recept_outer_p = get_receptance_outer(idx_base_sb + s)
                    
                    term_s_sb = (Pr_val / 2.0) * Fs_val * (
                        recept_inner_m + recept_inner_p + 
                        recept_outer_m + recept_outer_p
                    )
                    Y_sb_term2_sum += term_s_sb
                    
                Y_sb_total = Y_sb_term1 + Y_sb_term2_sum
                Acc_sb = abs(Y_sb_total * (w_sb_rad**2))
                
                results.append({
                    'Harmonic_j': j,
                    'Type': sb_type,
                    'Sideband_r': r_abs,
                    'Frequency_Hz': f_sb_hz,
                    'Amplitude_Accel_m_s2': Acc_sb
                })
                
    # ---------------------------------------------------------
    # 4. Finalização
    # ---------------------------------------------------------
    df = pd.DataFrame(results)
    df = df.sort_values(by='Frequency_Hz').reset_index(drop=True)
    return df

if __name__ == "__main__":
    # Example usage for testing
    print("Testing calculate_tandon_coefficients...")
    coeffs = calculate_tandon_coefficients(1.5, 1500, 'inner')
    print(f"Po: {coeffs['Po']:.4f}")

    print("\nTesting calcular_espectro_inner_completo...")
    df_test = calcular_espectro_inner_completo(0.1778, 1750, 2, 2, 20)
    print(df_test.head(5))

    print("\nTesting calcular_espectro_outer_race...")
    df_outer = calcular_espectro_outer_race(0.1778, 1750, 5)
    print(df_outer)

    print("\nTesting calcular_espectro_ball_completo...")
    df_ball = calcular_espectro_ball_completo(0.1778, 1750, 3, 2, 20)
    print(df_ball.head(10))
