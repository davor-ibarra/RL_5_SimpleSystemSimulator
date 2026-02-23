import pandas as pd
import numpy as np
import sys

def analyze():
    filepath = r"c:\Users\davor\Escritorio\DOCTORADO\Tesis\RL_5_SimpleSystemSimulator\results_history_CartPole\cart_pole\20260223_1024\summary.xlsx"
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return

    print("--- SIMULATION SUMMARY ANALYSIS ---")
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    
    # 1. Check for NaNs
    nans = df.isna().sum()
    nans = nans[nans > 0]
    if len(nans) > 0:
        print("\n[!] Columns with NaN values:")
        for col, count in nans.items():
            print(f"  - {col}: {count} NaNs")
    else:
        print("\n[OK] No NaN values found.")

    # 2. Check for constant columns (variance == 0 or single unique value)
    print("\n[!] Constant columns (no variation):")
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
            val = df[col].iloc[0] if len(df) > 0 else None
            print(f"  - {col}: always {val}")

    # 3. Check for extremely large or small values
    print("\n[!] Min/Max values of interest:")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if col not in constant_cols:
                _min, _max = df[col].min(), df[col].max()
                _mean = df[col].mean()
                if "error" in col.lower() or "angle" in col.lower() or "position" in col.lower() or "reward" in col.lower() or "action" in col.lower() or "u_" in col.lower():
                    print(f"  - {col}: Min = {_min:.4f}, Max = {_max:.4f}, Mean = {_mean:.4f}")

    # 4. Check specific incongruencies based on RL CartPole expectations
    print("\n[!] Specific Checks:")
    if "is_saturated_global" in df.columns and "u_total" in df.columns:
        # Check if saturated flag is consistent with u_total
        # Usually saturation is at some limit. Let's see max u_total
        max_u = df["u_total"].abs().max()
        print(f"  - Max |u_total| = {max_u:.4f}")
        sat_count = df["is_saturated_global"].sum()
        print(f"  - is_saturated_global count: {sat_count}")
    
    if "agent_1_epsilon" in df.columns:
        print(f"  - Epsilon min: {df['agent_1_epsilon'].min():.4f}, max: {df['agent_1_epsilon'].max():.4f}")
        
    print("\nDone.")

if __name__ == '__main__':
    analyze()
