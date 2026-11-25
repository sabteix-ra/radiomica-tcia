
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\user\radiomica-tcia")

DATA = BASE / "data" / "nrrd_checked"
OUTPUT = BASE / "results"

OUTPUT.mkdir(parents=True, exist_ok=True)

PACIENTE = "LUNG1-030"

ct = sitk.ReadImage(
    str(DATA / (PACIENTE + "_seg_candidate_ct.nrrd"))
)

mask = sitk.ReadImage(
    str(DATA / (PACIENTE + "_seg_candidate_mask.nrrd"))
)

# Verificar a geometria
if not (
    ct.GetSize() == mask.GetSize()
    and np.allclose(ct.GetSpacing(), mask.GetSpacing())
    and np.allclose(ct.GetOrigin(), mask.GetOrigin())
    and np.allclose(ct.GetDirection(), mask.GetDirection())
):
    raise ValueError("Geometria incompatível")

# Converter para arrays
ct_array = sitk.GetArrayFromImage(ct)
mask_array = sitk.GetArrayFromImage(mask)

# Selecionar apenas os voxels tumorais
tumor = ct_array[mask_array == 1]

if tumor.size == 0:
    raise ValueError("Máscara vazia")

# Criar histograma
plt.figure(figsize=(8, 5))

plt.hist(
    tumor,
    bins=80,
    color="steelblue",
    edgecolor="black",
    linewidth=0.3
)

plt.axvline(
    np.mean(tumor),
    color="red",
    linestyle="--",
    label="Média"
)

plt.axvline(
    np.median(tumor),
    color="green",
    linestyle="--",
    label="Mediana"
)

plt.xlabel("Intensidade (HU)")
plt.ylabel("Número de voxels")
plt.title("Distribuição das intensidades tumorais - " + PACIENTE)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT / "LUNG1-030_histograma.png",
    dpi=300
)

plt.show()
