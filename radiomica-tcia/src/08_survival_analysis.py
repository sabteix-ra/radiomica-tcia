from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

# Diretorios
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "clinical"
OUT = BASE / "results" / "survival"

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

T = df["survival_days"]
E = df["event"]

print("\nANALISE DE SOBREVIVENCIA")
print("Pacientes:", len(df))
print("Obitos:", E.sum())
print("Censurados:", len(df) - E.sum())

# ==========================================
# 1. KAPLAN-MEIER GLOBAL
# ==========================================

kmf = KaplanMeierFitter()

kmf.fit(
    T,
    event_observed=E,
    label="Coorte completa"
)

print("\nKAPLAN-MEIER GLOBAL")

print(
    "Sobrevivencia mediana (dias):",
    kmf.median_survival_time_
)

for day in [365, 730, 1825]:

    probability = kmf.predict(day)

    print(
        f"Sobrevivencia aos {day} dias:",
        round(probability * 100, 2), "%"
    )

kmf.survival_function_.to_csv(
    OUT / "kaplan_meier_global.csv"
)

kmf.event_table.to_csv(
    OUT / "survival_event_table.csv"
)

plt.figure(figsize=(8, 6))

ax = kmf.plot_survival_function(
    ci_show=True,
    show_censors=True
)

ax.set_xlabel("Tempo (dias)")
ax.set_ylabel("Probabilidade de sobrevivencia")
ax.set_title("Sobrevivencia global - Kaplan-Meier")
ax.set_ylim(0, 1.05)

plt.tight_layout()

plt.savefig(
    OUT / "kaplan_meier_global.png",
    dpi=300
)

plt.close()

# ==========================================
# 2. KAPLAN-MEIER POR CARACTERISTICA
# ==========================================

logrank_results = []

for feature in features:

    median = df[feature].median()

    low = df[df[feature] <= median]
    high = df[df[feature] > median]

    if low.empty or high.empty:
        continue

    km_low = KaplanMeierFitter()
    km_high = KaplanMeierFitter()

    km_low.fit(
        low["survival_days"],
        low["event"],
        label=f"{feature} baixo (n={len(low)})"
    )

    km_high.fit(
        high["survival_days"],
        high["event"],
        label=f"{feature} alto (n={len(high)})"
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    km_low.plot_survival_function(
        ax=ax,
        ci_show=True,
        show_censors=True
    )

    km_high.plot_survival_function(
        ax=ax,
        ci_show=True,
        show_censors=True
    )

    # Teste log-rank
    result = logrank_test(
        low["survival_days"],
        high["survival_days"],
        event_observed_A=low["event"],
        event_observed_B=high["event"]
    )

    logrank_results.append({
        "feature": feature,
        "median_cutoff": median,
        "n_low": len(low),
        "n_high": len(high),
        "events_low": low["event"].sum(),
        "events_high": high["event"].sum(),
        "median_survival_low":
            km_low.median_survival_time_,
        "median_survival_high":
            km_high.median_survival_time_,
        "p_value": result.p_value
    })

    ax.set_title(
        f"{feature} - Log-rank p={result.p_value:.3f}"
    )

    ax.set_xlabel("Tempo (dias)")
    ax.set_ylabel("Probabilidade de sobrevivencia")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    plt.savefig(
        OUT / f"kaplan_meier_{feature}.png",
        dpi=300
    )

    plt.close()

logrank_df = pd.DataFrame(logrank_results)

logrank_df.to_csv(
    OUT / "logrank_results.csv",
    index=False
)

print("\nRESULTADOS LOG-RANK")
print(logrank_df.to_string(index=False))

# ==========================================
# 3. REGRESSAO DE COX UNIVARIADA
# ==========================================

cox_results = []

for feature in features:

    data = df[
        ["survival_days", "event", feature]
    ].copy()

    # Padronizacao da caracteristica
    mean = data[feature].mean()
    std = data[feature].std()

    if std == 0:
        continue

    data[feature] = (
        data[feature] - mean
    ) / std

    cph = CoxPHFitter()

    try:

        cph.fit(
            data,
            duration_col="survival_days",
            event_col="event"
        )

        summary = cph.summary.loc[feature]

        cox_results.append({
            "feature": feature,
            "coefficient": summary["coef"],
            "hazard_ratio": summary["exp(coef)"],
            "CI_lower":
                summary["exp(coef) lower 95%"],
            "CI_upper":
                summary["exp(coef) upper 95%"],
            "p_value": summary["p"],
            "c_index_apparent":
                cph.concordance_index_,
            "status": "OK"
        })

    except Exception as error:

        print("Erro Cox:", feature, error)

        cox_results.append({
            "feature": feature,
            "status": str(error)
        })

cox_df = pd.DataFrame(cox_results)

cox_df.to_csv(
    OUT / "cox_univariate.csv",
    index=False
)

print("\nREGRESSAO DE COX UNIVARIADA")
print(cox_df.to_string(index=False))

# ==========================================
# 4. FOREST PLOT
# ==========================================

valid = cox_df[
    cox_df["status"] == "OK"
].copy()

if not valid.empty:

    valid = valid.reset_index(drop=True)

    hr = valid["hazard_ratio"]

    lower = hr - valid["CI_lower"]
    upper = valid["CI_upper"] - hr

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        hr,
        np.arange(len(valid)),
        xerr=[lower, upper],
        fmt="o",
        capsize=5
    )

    ax.axvline(
        1,
        linestyle="--",
        color="black"
    )

    ax.set_yticks(np.arange(len(valid)))
    ax.set_yticklabels(valid["feature"])

    ax.set_xscale("log")

    ax.set_xlabel(
        "Hazard Ratio por aumento de 1 DP"
    )

    ax.set_title("Regressao de Cox univariada")

    plt.tight_layout()

    plt.savefig(
        OUT / "cox_forest_plot.png",
        dpi=300
    )

    plt.close()

print("\nANALISE TERMINADA")
print("Resultados guardados em:", OUT)