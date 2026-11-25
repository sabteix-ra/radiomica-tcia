from pathlib import Path
import pandas as pd
import numpy as np

# Diretorios
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
ANALYSIS = BASE / "results" / "analysis"
OUT = DATA / "clinical"

OUT.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. LEITURA DOS DADOS
# ==========================================

clinical = pd.read_csv(
    OUT / "Lung1.clinical.csv"
)

radiomics = pd.read_csv(
    DATA / "features" / "features_primary.csv"
)

selected = pd.read_csv(
    ANALYSIS / "selected_features.csv"
)

clinical.columns = clinical.columns.str.strip()

print("\nDADOS CLINICOS")
print("Pacientes:", len(clinical))
print("Colunas:", clinical.columns.tolist())

# ==========================================
# 2. NORMALIZACAO DAS COLUNAS
# ==========================================

clinical = clinical.rename(columns={
    "PatientID": "patient",
    "Survival.time": "survival_days",
    "deadstatus.event": "event"
})

required = [
    "patient",
    "survival_days",
    "event"
]

for col in required:
    if col not in clinical.columns:
        raise ValueError(
            f"Coluna em falta: {col}"
        )

for table in [clinical, radiomics, selected]:

    if table["patient"].isna().any():
        raise ValueError("Identificador em falta")

    table["patient"] = (
        table["patient"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

# ==========================================
# 3. SELECAO DOS DEZ PACIENTES
# ==========================================

patients = radiomics["patient"].tolist()

clinical_subset = clinical[
    clinical["patient"].isin(patients)
].copy()

print("\nPACIENTES CLINICOS ENCONTRADOS")
print(len(clinical_subset))

missing = set(patients) - set(
    clinical_subset["patient"]
)

if missing:
    raise ValueError(
        f"Pacientes em falta: {missing}"
    )

if clinical_subset["patient"].duplicated().any():
    raise ValueError("Pacientes duplicados")

if len(patients) != 10 or len(set(patients)) != 10:
    raise ValueError("Tabela radiomica invalida")

# ==========================================
# 4. VALIDACAO DA SOBREVIVENCIA
# ==========================================

for col in ["survival_days", "event"]:

    clinical_subset[col] = pd.to_numeric(
        clinical_subset[col],
        errors="coerce"
    )

    if not np.isfinite(clinical_subset[col]).all():
        raise ValueError(
            f"Valores invalidos em {col}"
        )

if (clinical_subset["survival_days"] < 0).any():
    raise ValueError("Tempos negativos")

if not clinical_subset["event"].isin([0, 1]).all():
    raise ValueError("Estado vital invalido")

clinical_subset["event"] = (
    clinical_subset["event"].astype(int)
)

clinical_subset["survival_months"] = (
    clinical_subset["survival_days"] / 30.4375
)

# ==========================================
# 5. INTEGRACAO DOS DADOS
# ==========================================

if set(selected["patient"]) != set(patients):
    raise ValueError("Pacientes inconsistentes")

merged = selected.merge(
    clinical_subset,
    on="patient",
    how="inner",
    validate="one_to_one"
)

merged_all = radiomics.merge(
    clinical_subset,
    on="patient",
    how="inner",
    validate="one_to_one"
)

if len(merged) != 10 or len(merged_all) != 10:
    raise ValueError("Integracao incompleta")

# ==========================================
# 6. ESTATISTICA CLINICA DESCRITIVA
# ==========================================

print("\nDADOS INTEGRADOS")

columns = [
    "patient",
    "survival_days",
    "survival_months",
    "event",
    "Volume",
    "Sphericity",
    "Mean",
    "Entropy",
    "Contrast"
]

print(merged[columns].to_string(index=False))

n = len(merged)
deaths = merged["event"].sum()
censored = n - deaths

print("\nRESUMO CLINICO")
print("Pacientes:", n)
print("Obitos:", deaths)
print("Censurados:", censored)

print("\nTEMPO DE OBSERVACAO (DIAS)")
print(merged["survival_days"].describe())

# ==========================================
# 7. GUARDAR OS RESULTADOS
# ==========================================

clinical_subset.to_csv(
    OUT / "clinical_subset.csv",
    index=False
)

merged.to_csv(
    OUT / "radiomics_clinical.csv",
    index=False
)

merged_all.to_csv(
    OUT / "radiomics_clinical_all.csv",
    index=False
)

print("\nINTEGRACAO TERMINADA")
print("Resultados guardados em:", OUT)