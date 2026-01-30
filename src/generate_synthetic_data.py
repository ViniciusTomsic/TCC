import numpy as np
import pandas as pd
import random
import sys
import os
from typing import Dict, List, Optional, Sequence, Tuple

# Add src to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bearing_utils as bu

# =============================================================================
# CONFIGURATION
# =============================================================================
DIAMETERS = [0.5, 1.0, 1.2]
K_VALUES = {
    'inner': [1.0], 
    'outer': [1.0], 
    'ball': [1.0]
}
RPMS = [1730, 1750, 1772, 1797]
FS = 12000  # Sampling rate
NUM_RANDOM_SEGMENTS = 10  # Number of random normal segments to use as baselines
MAX_S_ITER = 50 # Reverted number of force modes sum

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

_FAULT_PT_TO_EN = {
    "Pista Externa": "outer",
    "Pista Interna": "inner",
    "Esfera": "ball",
    "Outer": "outer",
    "Inner": "inner",
    "Ball": "ball",
    "OR": "outer",
    "IR": "inner",
    "B": "ball",
    "outer": "outer",
    "inner": "inner",
    "ball": "ball",
}

_FAULT_EN_TO_PT = {
    "outer": "Pista Externa",
    "inner": "Pista Interna",
    "ball": "Esfera",
}


def _normalize_fault_type(tipo_falha: str) -> Tuple[str, str]:
    """Normaliza tipo de falha e retorna (en, pt)."""
    if tipo_falha is None:
        raise ValueError("tipo_falha não pode ser None")
    tipo_falha_str = str(tipo_falha).strip()
    en = _FAULT_PT_TO_EN.get(tipo_falha_str)
    if en is None:
        raise ValueError(
            f"tipo_falha inválido: {tipo_falha!r}. Use 'Pista Externa'|'Pista Interna'|'Esfera' ou 'outer'|'inner'|'ball'."
        )
    pt = _FAULT_EN_TO_PT[en]
    return en, pt


def pad_or_trim(sig: np.ndarray, n_points: int) -> np.ndarray:
    """Garante comprimento `n_points` (corta ou completa com zeros)."""
    if len(sig) > n_points:
        return sig[:n_points]
    if len(sig) < n_points:
        return np.pad(sig, (0, n_points - len(sig)))
    return sig


def synthesize_time_signal(spectrum_df, duration=1.0, fs=FS):
    """
    Reconstructs time domain signal from spectrum df (freq, amp).
    Uses vectorized operations for speed.
    """
    if spectrum_df.empty:
        return np.zeros(int(duration * fs))
        
    t = np.arange(0, duration, 1/fs)
    
    # Extract arrays and reshape for broadcasting
    # Frequencies shape: (N, 1)
    freqs = spectrum_df['Frequency_Hz'].values.reshape(-1, 1)
    
    # Amplitudes shape: (N, 1)
    # NOTE: Using 'Amplitude_Accel_m_s2' as standardized in bearing_utils
    amps = spectrum_df['Amplitude_Accel_m_s2'].values.reshape(-1, 1)
    
    # Time shape: (1, T)
    t_grid = t.reshape(1, -1)
    
    # Cosine argument: 2*pi * f * t
    # Result shape: (N, T)
    cos_components = np.cos(2 * np.pi * freqs * t_grid)
    
    # Weighted sum
    # Result shape: (N, T) -> sum -> (T,)
    signal = np.sum(amps * cos_components, axis=0)
        
    return signal


def calcular_espectro_tandon(
    *,
    diametro_mm: float,
    rpm: int,
    tipo_falha: str,
    k_val: float,
    max_harmonics: int = 7,
    num_sidebands: int = 5,
    max_s_iter: int = MAX_S_ITER,
):
    """
    Calcula espectro (DataFrame) via modelagem de Tandon, usando funções de `bearing_utils`.
    Retorna DataFrame com colunas `Frequency_Hz`, `Amplitude_Accel_m_s2`, `K`.
    """
    fault_en, _ = _normalize_fault_type(tipo_falha)
    if fault_en == "outer":
        return bu.calcular_espectro_outer_race(
            diametro_mm,
            rpm,
            max_harmonics=max_harmonics,
            K=k_val,
        )
    if fault_en == "inner":
        return bu.calcular_espectro_inner_completo(
            diametro_mm,
            rpm,
            max_harmonics=max_harmonics,
            num_sidebands=num_sidebands,
            max_s_iter=max_s_iter,
            K=k_val,
        )
    return bu.calcular_espectro_ball_completo(
        diametro_mm,
        rpm,
        max_harmonics=max_harmonics,
        num_sidebands=num_sidebands,
        max_s_iter=max_s_iter,
        K=k_val,
    )


def gerar_sinal_tandon_puro(
    *,
    fs: float,
    n_points: int,
    diametro_mm: float,
    rpm: int,
    tipo_falha: str,
    k_val: float,
    max_harmonics: int = 7,
    num_sidebands: int = 5,
    max_s_iter: int = MAX_S_ITER,
) -> np.ndarray:
    """Gera sinal 'puro' (falha) no tempo, a partir do espectro Tandon."""
    spec_df = calcular_espectro_tandon(
        diametro_mm=diametro_mm,
        rpm=rpm,
        tipo_falha=tipo_falha,
        k_val=k_val,
        max_harmonics=max_harmonics,
        num_sidebands=num_sidebands,
        max_s_iter=max_s_iter,
    )
    sig = synthesize_time_signal(spec_df, duration=n_points / fs, fs=fs)
    return pad_or_trim(sig, n_points)


def gerar_sinal_tandon_completo(
    *,
    fs: float,
    n_points: int,
    diametro_mm: float,
    rpm: int,
    tipo_falha: str,
    k_val: float,
    sinal_normal: np.ndarray,
    max_harmonics: int = 7,
    num_sidebands: int = 5,
    max_s_iter: int = MAX_S_ITER,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna (sinal_puro, sinal_final) onde:
    - sinal_puro: falha via Tandon
    - sinal_final: sinal_normal + sinal_puro
    """
    base = pad_or_trim(np.asarray(sinal_normal), n_points)
    puro = gerar_sinal_tandon_puro(
        fs=fs,
        n_points=n_points,
        diametro_mm=diametro_mm,
        rpm=rpm,
        tipo_falha=tipo_falha,
        k_val=k_val,
        max_harmonics=max_harmonics,
        num_sidebands=num_sidebands,
        max_s_iter=max_s_iter,
    )
    return puro, base + puro


def gerar_dados_sinteticos_tandon_df(
    *,
    dicionario_treino: Dict[str, "object"],
    fs: float = FS,
    rpms: Sequence[int] = tuple(RPMS),
    diametros_mm: Sequence[float] = tuple(DIAMETERS),
    k_values: Optional[Dict[str, Sequence[float]]] = None,
    num_random_segments: int = NUM_RANDOM_SEGMENTS,
    seed: Optional[int] = None,
    incluir_normais_reais: bool = True,
    max_harmonics: int = 7,
    num_sidebands: int = 5,
    max_s_iter: int = MAX_S_ITER,
) -> pd.DataFrame:
    """
    Gera um DataFrame (schema impulse-like) com sinais via Tandon.

    Colunas:
    - rpm
    - tipo_falha_adicionada  ('Normal'|'Pista Externa'|'Pista Interna'|'Esfera')
    - diametro_falha_mm      (float ou NaN)
    - k_val                 (float ou NaN)
    - sinal_puro            (np.ndarray)
    - sinal_final           (np.ndarray)
    - metodo                ('real_normal'|'tandon')
    - base_normal           (chave do segmento usado como baseline)
    """
    rng = random.Random(seed)
    if k_values is None:
        k_values = K_VALUES

    rows: List[dict] = []

    if incluir_normais_reais:
        for key, df_seg in dicionario_treino.items():
            try:
                if str(df_seg["tipo_falha"].iloc[0]) != "Normal":
                    continue
                rpm = int(df_seg["rotacao_rpm"].iloc[0])
                sig = df_seg["amplitude"].values
                rows.append(
                    {
                        "rpm": rpm,
                        "tipo_falha_adicionada": "Normal",
                        "diametro_falha_mm": np.nan,
                        "k_val": np.nan,
                        "sinal_puro": np.zeros_like(sig),
                        "sinal_final": sig,
                        "metodo": "real_normal",
                        "base_normal": key,
                    }
                )
            except Exception:
                continue

    for rpm in rpms:
        rpm_candidates: List[Tuple[str, np.ndarray]] = []
        for key, df_seg in dicionario_treino.items():
            try:
                if str(df_seg["tipo_falha"].iloc[0]) != "Normal":
                    continue
                if int(df_seg["rotacao_rpm"].iloc[0]) != int(rpm):
                    continue
                rpm_candidates.append((key, df_seg["amplitude"].values))
            except Exception:
                continue

        if not rpm_candidates:
            continue

        baselines = (
            rng.sample(rpm_candidates, num_random_segments)
            if len(rpm_candidates) > num_random_segments
            else rpm_candidates
        )

        for base_key, base_sig in baselines:
            n_points = len(base_sig)
            base_sig = np.asarray(base_sig)

            for fault_en in ["inner", "outer", "ball"]:
                fault_pt = _FAULT_EN_TO_PT[fault_en]
                for k_val in k_values.get(fault_en, []):
                    for diam in diametros_mm:
                        puro, final = gerar_sinal_tandon_completo(
                            fs=fs,
                            n_points=n_points,
                            diametro_mm=float(diam),
                            rpm=int(rpm),
                            tipo_falha=fault_en,
                            k_val=float(k_val),
                            sinal_normal=base_sig,
                            max_harmonics=max_harmonics,
                            num_sidebands=num_sidebands,
                            max_s_iter=max_s_iter,
                        )
                        rows.append(
                            {
                                "rpm": int(rpm),
                                "tipo_falha_adicionada": fault_pt,
                                "diametro_falha_mm": float(diam),
                                "k_val": float(k_val),
                                "sinal_puro": puro,
                                "sinal_final": final,
                                "metodo": "tandon",
                                "base_normal": base_key,
                            }
                        )

    return pd.DataFrame(rows)

# =============================================================================
# MAIN GENERATION LOGIC
# =============================================================================

def main():
    # Import aqui para evitar side-effects ao importar este módulo
    import segment_and_split_data as ssd

    print("Starting synthetic data generation (Tandon)...")
    df = gerar_dados_sinteticos_tandon_df(
        dicionario_treino=ssd.dicionario_treino,
        fs=FS,
        rpms=RPMS,
        diametros_mm=DIAMETERS,
        k_values=K_VALUES,
        num_random_segments=NUM_RANDOM_SEGMENTS,
        max_s_iter=MAX_S_ITER,
    )

    print("\nGeneration Complete.")
    print(f"Total records in final DataFrame: {len(df)}")
    if "tipo_falha_adicionada" in df.columns:
        print("\nClass Distribution:")
        print(df["tipo_falha_adicionada"].value_counts())
    print("\nSample Data:")
    print(df.head())
    return df

if __name__ == "__main__":
    df = main()
