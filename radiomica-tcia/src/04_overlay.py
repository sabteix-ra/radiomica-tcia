
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# 1. Caminhos

BASE = Path(r"C:\Users\user\radiomica-tcia")

#Para LUNG_030 DATA = BASE / "data" / "nrrd_checked"
DATA = BASE / "data" / "nrrd"
OUTPUT = BASE / "results"

OUTPUT.mkdir(parents=True, exist_ok=True)

PACIENTE = "LUNG1-100"

# 2. Carregar a TC e a máscara

ct = sitk.ReadImage(
    str(DATA / (PACIENTE + "_ct.nrrd"))
)

mask = sitk.ReadImage(
    str(DATA / (PACIENTE + "_gtv_dicomseg.nrrd"))
)

# 3. Verificar a geometria

if not (
    ct.GetSize() == mask.GetSize()
    and np.allclose(ct.GetSpacing(), mask.GetSpacing())
    and np.allclose(ct.GetOrigin(), mask.GetOrigin())
    and np.allclose(ct.GetDirection(), mask.GetDirection())
):
    raise ValueError("Geometria incompatível")

# Orientar os volumes segundo os eixos anatómicos

ct = sitk.DICOMOrient(ct, "LPS")
mask = sitk.DICOMOrient(mask, "LPS")

# 4. Converter para arrays

imagem = sitk.GetArrayFromImage(ct)
mascara = sitk.GetArrayFromImage(mask) == 1

sx, sy, sz = ct.GetSpacing()

# 5. Determinar o centro da máscara

coords = np.argwhere(mascara)

if len(coords) == 0:
    raise ValueError("Máscara vazia")

z, y, x = np.round(
    np.mean(coords, axis=0)
).astype(int)

print("Centro da máscara (z, y, x):", z, y, x)

# 6. Selecionar os três planos

planos = [
    (imagem[z, :, :], mascara[z, :, :],
     sy / sx, "Axial", "upper"),

    (imagem[:, y, :], mascara[:, y, :],
     sz / sx, "Coronal", "lower"),

    (imagem[:, :, x], mascara[:, :, x],
     sz / sy, "Sagital", "lower")
]

# 7. Criar a figura

fig, axes = plt.subplots(
    1, 3, figsize=(15, 6),
    facecolor="black"
)

for ax, plano in zip(axes, planos):

    img, roi, proporcao, titulo, origem = plano

    # Visualizar a TC em escala de cinzentos
    ax.imshow(
    img,
    cmap="gray",
    vmin=-160,
    vmax=240,
    aspect=proporcao,
    origin=origem
)

    # Sobrepor a máscara tumoral
    overlay = np.ma.masked_where(~roi, roi)

    ax.imshow(
        overlay,
        cmap="autumn",
        vmin=0,
        vmax=1,
        alpha=0.35,
        aspect=proporcao,
        origin=origem
    )

    # Contorno da segmentação
    if roi.any() and not roi.all():

        linhas, colunas = roi.shape

        X, Y = np.meshgrid(
            np.arange(colunas),
            np.arange(linhas)
    )

    ax.contour(
        X, Y, roi.astype(float),
        levels=[0.5],
        colors=["red"],
        linewidths=0.8
    )

    ax.set_title(
        titulo,
        color="white",
        fontsize=14
    )

    ax.axis("off")

fig.suptitle(
    "Segmentação tumoral - " + PACIENTE,
    color="white",
    fontsize=17
)

plt.tight_layout()

# 8. Guardar a figura

destino = OUTPUT / (PACIENTE + "_overlay.png")

plt.savefig(
    destino,
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

print("Figura guardada em:", destino)

plt.show()
