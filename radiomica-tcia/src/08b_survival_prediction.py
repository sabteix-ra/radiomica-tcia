from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# Diretorios
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "clinical"
OUT = BASE / "results" / "prediction"

OUT.mkdir(parents=True, exist_ok=True)

# Leitura dos dados
df = pd.read_csv(
    DATA / "radiomics_clinical.csv"
)

features = [
    "Volume",
    "Sphericity",
    "Mean",
    "Entropy",
    "Contrast"
]

# Horizonte temporal em dias
times = [365, 730, 1095]

predictions = []
results = []

# ==========================================
# 1. VALIDACAO DOS DADOS
# ==========================================

if len(df) != 10:
    raise ValueError("Numero inesperado de pacientes")

if df["patient"].duplicated().any():
    raise ValueError("Pacientes duplicados")

print("\nVALIDACAO CRUZADA LEAVE-ONE-OUT")
print("Pacientes:", len(df))
print("Caracteristicas:", len(features))

# ==========================================
# 2. MODELOS UNIVARIADOS
# ==========================================

for feature in features:

    print("\nCaracteristica:", feature)

    data = df[
        ["patient", "survival_days", "event", feature]
    ].copy()

    # ======================================
    # 3. MODELO COM TODOS OS PACIENTES
    # ======================================

    full = data.drop(columns="patient").copy()

    full[feature] = (
        full[feature] - full[feature].mean()
    ) / full[feature].std()

    full_model = CoxPHFitter()

    full_model.fit(
        full,
        duration_col="survival_days",
        event_col="event"
    )

    apparent_cindex = full_model.concordance_index_

    # ======================================
    # 4. LEAVE-ONE-OUT
    # ======================================

    feature_predictions = []

    for i in range(len(data)):

        train = data.drop(
            index=data.index[i]
        ).copy()

        test = data.iloc[[i]].copy()

        patient = test["patient"].iloc[0]

        # Padronizacao apenas com dados de treino
        mean = train[feature].mean()
        std = train[feature].std()

        if std == 0:
            raise ValueError(
                "Caracteristica constante no treino"
            )

        train[feature] = (
            train[feature] - mean
        ) / std

        test[feature] = (
            test[feature] - mean
        ) / std

        train_model = train.drop(
            columns="patient"
        )

        test_model = test[[feature]]

        model = CoxPHFitter()

        try:

            model.fit(
                train_model,
                duration_col="survival_days",
                event_col="event"
            )

            # Previsao de sobrevivencia
            survival = model.predict_survival_function(
                test_model,
                times=times
            )

            s12 = float(survival.iloc[0, 0])
            s24 = float(survival.iloc[1, 0])
            s36 = float(survival.iloc[2, 0])

            row = {
                "patient": patient,
                "feature": feature,
                "survival_days":
                    test["survival_days"].iloc[0],
                "event":
                    test["event"].iloc[0],
                "survival_12m": s12,
                "survival_24m": s24,
                "survival_36m": s36,
                "risk_12m": 1 - s12,
                "coefficient":
                    model.params_.iloc[0],
                "status": "OK"
            }

            predictions.append(row)
            feature_predictions.append(row)

            print(
                patient,
                "S(12 meses):",
                round(s12, 4)
            )

        except Exception as error:

            print("ERRO:", patient, error)

            predictions.append({
                "patient": patient,
                "feature": feature,
                "status": str(error)
            })

    # ======================================
    # 5. C-INDEX VALIDADO
    # ======================================

    if len(feature_predictions) != len(data):

        print("Validacao incompleta:", feature)

        results.append({
            "feature": feature,
            "c_index_apparent": apparent_cindex,
            "c_index_loocv": np.nan,
            "status": "INCOMPLETE"
        })

        continue

    pred = pd.DataFrame(feature_predictions)

    cv_cindex = concordance_index(
        pred["survival_days"],
        pred["survival_12m"],
        pred["event"]
    )

    results.append({
        "feature": feature,
        "c_index_apparent": apparent_cindex,
        "c_index_loocv": cv_cindex,
        "difference": apparent_cindex - cv_cindex,
        "status": "OK"
    })

    print(
        "C-index aparente:",
        round(apparent_cindex, 4)
    )

    print(
        "C-index LOOCV:",
        round(cv_cindex, 4)
    )

# ==========================================
# 6. GUARDAR RESULTADOS
# ==========================================

predictions_df = pd.DataFrame(predictions)
results_df = pd.DataFrame(results)

predictions_df.to_csv(
    OUT / "loocv_predictions.csv",
    index=False
)

results_df.to_csv(
    OUT / "loocv_results.csv",
    index=False
)

print("\nRESULTADOS FINAIS")
print(results_df.to_string(index=False))

# ==========================================
# 7. COMPARACAO DOS C-INDEX
# ==========================================

x = np.arange(len(results_df))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    x - width / 2,
    results_df["c_index_apparent"],
    width,
    label="Aparente"
)

ax.bar(
    x + width / 2,
    results_df["c_index_loocv"],
    width,
    label="LOOCV"
)

ax.axhline(
    0.5,
    color="black",
    linestyle="--",
    label="Referencia aleatoria"
)

ax.set_xticks(x)
ax.set_xticklabels(results_df["feature"])
ax.set_ylim(0, 1)
ax.set_ylabel("C-index")
ax.set_title("Avaliacao da capacidade prognostica")
ax.legend()

plt.tight_layout()

plt.savefig(
    OUT / "cindex_comparison.png",
    dpi=300
)

plt.close()

# ==========================================
# 8. PREVISAO INDIVIDUAL - VOLUME
# ==========================================

volume = predictions_df[
    (predictions_df["feature"] == "Volume") &
    (predictions_df["status"] == "OK")
].copy()

if not volume.empty:

    volume = volume.sort_values("survival_12m")

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(volume))
    width = 0.25

    ax.bar(
        x - width,
        volume["survival_12m"],
        width,
        label="12 meses"
    )

    ax.bar(
        x,
        volume["survival_24m"],
        width,
        label="24 meses"
    )

    ax.bar(
        x + width,
        volume["survival_36m"],
        width,
        label="36 meses"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        volume["patient"],
        rotation=45
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilidade de sobrevivencia")
    ax.set_title("Previsao individual - Volume tumoral")
    ax.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "individual_survival_prediction.png",
        dpi=300
    )

    plt.close()

print("\nVALIDACAO TERMINADA")
print("Resultados guardados em:", OUT)