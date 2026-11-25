
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\user\radiomica-tcia")

DATA = BASE / "data/nrrd"
OUTPUT = BASE / "results/visual_validation"

OUTPUT.mkdir(parents=True, exist_ok=True)

pacientes = [
    "LUNG1-" + str(i).zfill(3)
    for i in range(10, 101, 10)
]

for paciente in pacientes:

    ct = sitk.ReadImage(
        str(DATA / (paciente + "_ct.nrrd"))
    )

    ct = sitk.DICOMOrient(ct, "LPS")

    imagem = sitk.GetArrayFromImage(ct)

    sx, sy, sz = ct.GetSpacing()

    for tipo in ["gtv_ref", "gtv_dicomseg"]:

        caminho = DATA / (
            paciente + "_" + tipo + ".nrrd"
        )

        if not caminho.exists():
            continue

        mask = sitk.ReadImage(str(caminho))

        mask = sitk.DICOMOrient(mask, "LPS")

        roi = sitk.GetArrayFromImage(mask) == 1

        if not np.any(roi):
            print("Mascara vazia:", paciente, tipo)
            continue

        # Corte com maior area segmentada
        # em cada plano anatomico

        z = np.argmax(roi.sum(axis=(1, 2)))
        y = np.argmax(roi.sum(axis=(0, 2)))
        x = np.argmax(roi.sum(axis=(0, 1)))

        cortes = [
            (
                imagem[z, :, :],
                roi[z, :, :],
                "Axial",
                sy / sx,
                "upper"
            ),
            (
                imagem[:, y, :],
                roi[:, y, :],
                "Coronal",
                sz / sx,
                "lower"
            ),
            (
                imagem[:, :, x],
                roi[:, :, x],
                "Sagital",
                sz / sy,
                "lower"
            )
        ]

        fig, axes = plt.subplots(
            2, 3, figsize=(15, 10)
        )

        janelas = [
            (-160, 240, "Tecidos moles"),
            (-1350, 150, "Pulmonar")
        ]

        for linha, janela in enumerate(janelas):

            vmin, vmax, nome = janela

            for coluna, corte in enumerate(cortes):

                img, regiao, plano, aspecto, origem = corte

                ax = axes[linha, coluna]

                ax.imshow(
                    img,
                    cmap="gray",
                    vmin=vmin,
                    vmax=vmax,
                    aspect=aspecto,
                    origin=origem
                )

                overlay = np.ma.masked_where(
                    ~regiao,
                    regiao.astype(float)
                )

                ax.imshow(
                    overlay,
                    cmap="autumn",
                    alpha=0.35,
                    aspect=aspecto,
                    origin=origem,
                    vmin=0,
                    vmax=1
                )

                ax.contour(
                    regiao.astype(float),
                    levels=[0.5],
                    colors=["red"],
                    linewidths=0.8
                )

                ax.set_title(plano + " - " + nome)

                ax.axis("off")

        fig.suptitle(paciente + " - " + tipo)

        plt.tight_layout()

        plt.savefig(
            OUTPUT / (
                paciente + "_" + tipo + ".png"
            ),
            dpi=200
        )

        plt.close()

        print("Imagem criada:", paciente, tipo)

print("\nInspecao visual terminada")
