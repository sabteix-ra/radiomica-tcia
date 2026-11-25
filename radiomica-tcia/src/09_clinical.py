
from pathlib import Path
import pandas as pd

BASE = Path(r"C:\Users\user\radiomica-tcia")

df = pd.read_csv(
    BASE / "clinical/clinical_selected.csv",
    sep=None,
    engine="python"
)

print("\n--- DADOS CLINICOS ---")
print(df.to_string(index=False))

print("\n--- COLUNAS ---")
print(df.columns.tolist())

print("\nNumero de pacientes:", len(df))

print("\n--- DADOS EM FALTA ---")
print(df.isna().sum())

print("\n--- ESTATISTICA DESCRITIVA ---")
print(df.describe())
