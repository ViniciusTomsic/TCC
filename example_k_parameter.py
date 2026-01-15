"""
Example showing how to use the K (pulse height) parameter in bearing analysis functions.

K represents the pulse height coefficient in Tandon's model:
- K=1.0 (default): Normalized analysis
- K>1.0: Higher amplitude pulses (more severe defects)
- K<1.0: Lower amplitude pulses (less severe defects)
"""

from bearing_utils import (
    calcular_espectro_inner_completo,
    calcular_espectro_outer_race,
    calcular_espectro_ball_completo
)

# Common parameters
fault_mm = 0.1778
rpm_val = 1750

# ============================================================================
# Example 1: Inner Race with different K values
# ============================================================================
print("=== INNER RACE - Effect of K parameter ===")

# Default K=1.0
df_inner_k1 = calcular_espectro_inner_completo(
    fault_diameter_mm=fault_mm,
    rpm=rpm_val,
    max_harmonics=3,
    num_sidebands=2,
    K=1.0  # Default
)

# Higher intensity K=2.0
df_inner_k2 = calcular_espectro_inner_completo(
    fault_diameter_mm=fault_mm,
    rpm=rpm_val,
    max_harmonics=3,
    num_sidebands=2,
    K=2.0  # Double pulse height
)

print(f"\nWith K=1.0 (first main peak): {df_inner_k1.loc[df_inner_k1['Type']=='Main'].iloc[0]['Amplitude_Accel_m_s2']:.6f}")
print(f"With K=2.0 (first main peak): {df_inner_k2.loc[df_inner_k2['Type']=='Main'].iloc[0]['Amplitude_Accel_m_s2']:.6f}")

# ============================================================================
# Example 2: Outer Race with K parameter
# ============================================================================
print("\n=== OUTER RACE - Effect of K parameter ===")

df_outer_k1 = calcular_espectro_outer_race(
    fault_diameter_mm=fault_mm,
    rpm=rpm_val,
    max_harmonics=3,
    K=1.5
)

print(df_outer_k1.head(3))

# ============================================================================
# Example 3: Ball Bearing with K parameter
# ============================================================================
print("\n=== BALL BEARING - Effect of K parameter ===")

df_ball_custom = calcular_espectro_ball_completo(
    fault_diameter_mm=fault_mm,
    rpm=rpm_val,
    max_harmonics=2,
    num_sidebands=2,
    K=0.8  # Lower pulse height
)

print(df_ball_custom.head(5))
