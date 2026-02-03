import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.signal
from scipy.fft import fft
from scipy.spatial.distance import cosine


def apply_antialiasing_filter(signal: np.ndarray, fs: float, cutoff_ratio: float = 0.4, order: int = 4) -> np.ndarray:
    """
    Applies a lowpass Butterworth filter for anti-aliasing.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal (1D).
    fs : float
        Sampling frequency in Hz.
    cutoff_ratio : float, optional
        Ratio of the cutoff frequency to the sampling frequency. 
        Nyquist is 0.5. Default is 0.4.
    order : int, optional
        Order of the Butterworth filter. Default is 4.
        
    Returns:
    --------
    np.ndarray
        Filtered signal.
    """
    nyquist = 0.5 * fs
    cutoff = cutoff_ratio * fs
    
    # Ensure cutoff is within Nyquist limit
    if cutoff >= nyquist:
        cutoff = 0.99 * nyquist
        
    b, a = scipy.signal.butter(order, cutoff, fs=fs, btype='low', analog=False)
    filtered_signal = scipy.signal.filtfilt(b, a, signal)
    return filtered_signal


def apply_lowpass_filter(signal: np.ndarray, fs: float, cutoff_freq: float = 1200.0, order: int = 4) -> np.ndarray:
    """
    Applies a lowpass Butterworth filter with a specific cutoff frequency.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal (1D).
    fs : float
        Sampling frequency in Hz.
    cutoff_freq : float, optional
        Cutoff frequency in Hz. Default is 1200.0 Hz.
    order : int, optional
        Order of the Butterworth filter. Default is 4.
        
    Returns:
    --------
    np.ndarray
        Filtered signal.
    """
    nyquist = 0.5 * fs
    
    # Ensure cutoff is within Nyquist limit
    if cutoff_freq >= nyquist:
        cutoff_freq = 0.99 * nyquist
        
    b, a = scipy.signal.butter(order, cutoff_freq, fs=fs, btype='low', analog=False)
    filtered_signal = scipy.signal.filtfilt(b, a, signal)
    return filtered_signal


def apply_hanning_window(signal: np.ndarray) -> np.ndarray:
    """
    Applies a Hanning window to the signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal (1D).
        
    Returns:
    --------
    np.ndarray
        Signal multiplied by the Hanning window.
    """
    window = np.hanning(len(signal))
    return signal * window


def limit_spectrum_frequency(freqs: np.ndarray, spectrum: np.ndarray, min_freq: float, max_freq: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Limits the frequency spectrum to a specific range [min_freq, max_freq].
    
    Parameters:
    -----------
    freqs : np.ndarray
        Array of frequencies.
    spectrum : np.ndarray
        Array of spectrum magnitudes (corresponding to freqs).
    min_freq : float
        Minimum frequency to include.
    max_freq : float
        Maximum frequency to include.
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (sliced_freqs, sliced_spectrum).
    """
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    return freqs[mask], spectrum[mask]


# Diâmetros de falha no dataset CWRU (mm) -> string usada na coluna `diametro_falha`
CWRU_DIA_MAP_MM_TO_STR = {
    0.1778: '0.007"',
    0.3556: '0.014"',
    0.5334: '0.021"',
    0.7112: '0.028"',
}

# Códigos usados no notebook -> nomes usados na coluna `tipo_falha`
FAULT_CODE_TO_LABEL = {
    "OR": "Pista Externa",
    "IR": "Pista Interna",
    "B": "Esfera",
    "Normal": "Normal",
}


def calcular_sam_graus(fft_ref: np.ndarray, fft_alvo: np.ndarray) -> float:
    """Calcula o cosseno do ângulo SAM entre dois espectros (magnitude ou complexo)."""
    if np.iscomplexobj(fft_ref):
        fft_ref = np.abs(fft_ref)
    if np.iscomplexobj(fft_alvo):
        fft_alvo = np.abs(fft_alvo)

    min_len = min(len(fft_ref), len(fft_alvo))
    dist = cosine(fft_ref[:min_len], fft_alvo[:min_len])
    sim = np.clip(1.0 - dist, -1.0, 1.0)
    return float(sim)


def get_mag_spectrum(sig: np.ndarray) -> np.ndarray:
    """Espectro unilateral de magnitude normalizado."""
    yf = fft(sig)
    n = len(sig)
    return (2.0 / n) * np.abs(yf[0 : n // 2])


def pad_or_trim(sig: np.ndarray, n_points: int) -> np.ndarray:
    """Garante comprimento `n_points` (corta ou completa com zeros)."""
    if len(sig) > n_points:
        return sig[:n_points]
    if len(sig) < n_points:
        return np.pad(sig, (0, n_points - len(sig)))
    return sig


def _closest_cwru_diameter_mm(target_dia_mm: float) -> float:
    return min(CWRU_DIA_MAP_MM_TO_STR.keys(), key=lambda k: abs(k - target_dia_mm))


def get_real_signals_metadata_batch(
    dicionario_teste: Dict[str, "object"],
    rpm: int,
    type_code: str,
    target_dia_mm: float,
    num_samples: int,
    *,
    rng: Optional[random.Random] = None,
) -> List[Tuple[str, np.ndarray]]:
    """
    Busca sinais reais do `dicionario_teste` (segment_and_split_data) por RPM/tipo/diâmetro.
    Retorna lista de (chave, sinal_amplitude).
    """
    if rng is None:
        rng = random

    closest_mm = _closest_cwru_diameter_mm(target_dia_mm) if type_code != "Normal" else 0.1778
    target_dia_str = CWRU_DIA_MAP_MM_TO_STR.get(closest_mm, str(target_dia_mm))

    target_type = FAULT_CODE_TO_LABEL.get(type_code, type_code)

    candidates: List[Tuple[str, np.ndarray]] = []
    for chave, df in dicionario_teste.items():
        if getattr(df, "empty", False):
            continue
        try:
            row_rpm = int(df["rotacao_rpm"].iloc[0])
            row_type = str(df["tipo_falha"].iloc[0])
            row_dia = str(df["diametro_falha"].iloc[0]).replace('"', "").strip()
            target_dia_clean = str(target_dia_str).replace('"', "").strip()

            if abs(row_rpm - rpm) < 50 and row_type == target_type:
                if target_type == "Normal" or row_dia == target_dia_clean:
                    candidates.append((chave, df["amplitude"].values))
        except Exception:
            continue

    if len(candidates) > num_samples:
        return rng.sample(candidates, num_samples)
    return candidates


def get_normal_signal(
    dicionario_teste: Dict[str, "object"],
    rpm: int,
    n_points: int,
    *,
    fallback_std: float = 0.01,
) -> np.ndarray:
    """Retorna 1 segmento Normal para o RPM (ou ruído gaussiano fallback) já ajustado em `n_points`."""
    normals = get_real_signals_metadata_batch(
        dicionario_teste=dicionario_teste,
        rpm=rpm,
        type_code="Normal",
        target_dia_mm=0.0,
        num_samples=1,
    )
    if normals:
        return pad_or_trim(normals[0][1], n_points)
    return np.random.normal(0, fallback_std, n_points)

