from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Diretorios
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "features"
OUT = BASE / "results" / "analysis"

OUT.mkdir(parents=True, exist_ok=True)

# Leitura dos dados
df = pd.read_csv(DATA / "features_primary.csv")
all_masks = pd.read_csv(DATA / "features_all.csv")
comparison = pd.read_csv(DATA / "mask_comparison.csv")

# Caracteristicas selecionadas
features = {
    "original_shape_MeshVolume": "Volume",
    "original_shape_Sphericity": "Sphericity",
    "original_firstorder_Mean": "Mean",
    "original_firstorder_Entropy": "Entropy",
    "original_glcm_Contrast": "Contrast"
}

# ==========================================
# 1. VALIDACAO DOS DADOS
# ==========================================

radiomics_cols = [
    col for col in df.columns
    if col.startswith(("original_", "wavelet-", "wavelet_", "log-"))
]

numeric = df[radiomics_cols]

print("\nVALIDACAO DOS DADOS")
print("Pacientes:", len(df))
print("Mascaras:", len(all_masks))
print("Caracteristicas:", len(radiomics_cols))
print("Valores NaN:", numeric.isna().sum().sum())
print("Valores infinitos:", np.isinf(numeric).sum().sum())

constant = numeric.columns[
    numeric.nunique(dropna=True) <= 1
]

print("Caracteristicas constantes:", len(constant))

if len(df) != 10 or df["patient"].duplicated().any():
    raise ValueError("Tabela principal invalida")

# ==========================================
# 2. SELECAO DAS CARACTERISTICAS
# ==========================================

selected = df[
    ["patient"] + list(features.keys())
].copy()

selected = selected.rename(columns=features)

# Converter mm3 para cm3
selected["Volume"] = selected["Volume"] / 1000

selected.to_csv(
    OUT / "selected_features.csv",
    index=False
)

print("\nCARACTERISTICAS SELECIONADAS")
print(selected.to_string(index=False))

# ==========================================
# 3. ESTATISTICA DESCRITIVA
# ==========================================

stats = selected.drop(columns="patient").describe().T

stats["IQR"] = stats["75%"] - stats["25%"]

stats.to_csv(OUT / "descriptive_statistics.csv")

print("\nESTATISTICA DESCRITIVA")
print(stats)

# ==========================================
# 4. BOXPLOTS
# ==========================================

labels = {
    "Volume": "Volume (cm3)",
    "Sphericity": "Esfericidade",
    "Mean": "Intensidade media (HU)",
    "Entropy": "Entropia",
    "Contrast": "Contraste GLCM"
}

for feature in features.values():

    plt.figure(figsize=(5, 6))

    sns.boxplot(y=selected[feature])
    sns.stripplot(
        y=selected[feature],
        color="black",
        jitter=0.12
    )

    plt.ylabel(labels[feature])
    plt.title(labels[feature])
    plt.tight_layout()

    plt.savefig(
        OUT / f"boxplot_{feature}.png",
        dpi=300
    )

    plt.close()

# ==========================================
# 5. VOLUME POR PACIENTE
# ==========================================

plt.figure(figsize=(10, 5))

plt.bar(
    selected["patient"],
    selected["Volume"]
)

plt.ylabel("Volume tumoral (cm3)")
plt.xticks(rotation=45)
plt.title("Volume tumoral por paciente")
plt.tight_layout()

plt.savefig(
    OUT / "tumor_volume.png",
    dpi=300
)

plt.close()

# ==========================================
# 6. CORRELACAO DE SPEARMAN
# ==========================================

corr = selected.drop(
    columns="patient"
).corr(method="spearman")

corr.to_csv(OUT / "spearman_correlation.csv")

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True
)

plt.title("Correlacao de Spearman")
plt.tight_layout()

plt.savefig(
    OUT / "spearman_heatmap.png",
    dpi=300
)

plt.close()

# ==========================================
# 7. COMPARACAO DAS MASCARAS
# ==========================================

print("\nCOMPARACAO DAS MASCARAS")
print(comparison.to_string(index=False))

print("\nDice medio:", comparison["dice"].mean())
print("Dice minimo:", comparison["dice"].min())
print("Dice maximo:", comparison["dice"].max())

plt.figure(figsize=(9, 5))

plt.bar(
    comparison["patient"],
    comparison["dice"]
)

plt.ylim(0, 1.05)
plt.ylabel("Coeficiente de Dice")
plt.xticks(rotation=45)
plt.title("Concordancia entre segmentacoes")
plt.tight_layout()

plt.savefig(
    OUT / "dice_comparison.png",
    dpi=300
)

plt.close()

# ==========================================
# 8. ROBUSTEZ DAS CARACTERISTICAS
# ==========================================

robustness = []

for patient in all_masks["patient"].unique():

    group = all_masks[
        all_masks["patient"] == patient
    ]

    ref = group[group["mask"] == "gtv_ref"]
    seg = group[group["mask"] == "gtv_dicomseg"]

    if len(ref) != 1 or len(seg) != 1:
        continue

    for original, name in features.items():

        A = float(ref[original].iloc[0])
        B = float(seg[original].iloc[0])

        absolute_difference = abs(B - A)

        denominator = abs(A) + abs(B)

        if denominator == 0:
            difference_pct = 0
        else:
            difference_pct = (
                200 * absolute_difference / denominator
            )

        robustness.append({
            "patient": patient,
            "feature": name,
            "reference": A,
            "dicomseg": B,
            "absolute_difference": absolute_difference,
            "symmetric_difference_pct": difference_pct
        })

robustness = pd.DataFrame(robustness)

robustness.to_csv(
    OUT / "feature_robustness.csv",
    index=False
)

summary = robustness.groupby(
    "feature"
)["symmetric_difference_pct"].agg(
    ["mean", "median", "max"]
)

summary.to_csv(
    OUT / "robustness_summary.csv"
)

print("\nROBUSTEZ DAS CARACTERISTICAS")
print(summary)

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=robustness,
    x="feature",
    y="symmetric_difference_pct"
)

sns.stripplot(
    data=robustness,
    x="feature",
    y="symmetric_difference_pct",
    color="black"
)

plt.ylabel("Diferenca percentual simetrica (%)")
plt.xlabel("Caracteristica")
plt.title("Sensibilidade a segmentacao")
plt.tight_layout()

plt.savefig(
    OUT / "feature_robustness.png",
    dpi=300
)

plt.close()

print("\nANALISE TERMINADA")
print("Resultados guardados em:", OUT)