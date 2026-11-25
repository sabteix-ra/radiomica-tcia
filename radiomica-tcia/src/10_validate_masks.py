
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\user\radiomica-tcia")

DATA = BASE / "data/nrrd"
OUTPUT = BASE / "results"

OUTPUT.mkdir(exist_ok=True)

pacientes = [
    "LUNG1-" + str(i).zfill(3)
    for i in range(10, 101, 10)
]

resultados = []

def mesma_geometria(img1, img2):

    return (
        img1.GetSize() == img2.GetSize()
        and np.allclose(
            img1.GetSpacing(), img2.GetSpacing()
        )
        and np.allclose(
            img1.GetOrigin(), img2.GetOrigin()
        )
        and np.allclose(
            img1.GetDirection(), img2.GetDirection()
        )
    )

for paciente in pacientes:

    print("\n---", paciente, "---")

    ct_path = DATA / (paciente + "_ct.nrrd")

    if not ct_path.exists():
        print("TC inexistente")
        continue

    ct = sitk.ReadImage(str(ct_path))

    mascaras = {}

    for tipo in ["gtv_ref", "gtv_dicomseg"]:

        caminho = DATA / (paciente + "_" + tipo + ".nrrd")

        if not caminho.exists():
            print(tipo, ": inexistente")
            continue

        mask = sitk.ReadImage(str(caminho))

        arr = sitk.GetArrayFromImage(mask)

        geometria = mesma_geometria(ct, mask)

        valores = np.unique(arr)

        binaria = np.array_equal(valores, [0, 1])

        voxels = np.count_nonzero(arr == 1)

        volume = (
            voxels * np.prod(mask.GetSpacing()) / 1000
        )

        print("Mascara:", tipo)
        print("Geometria compativel:", geometria)
        print("Valores:", valores)
        print("Mascara binaria:", binaria)
        print("Voxels:", voxels)
        print("Volume (cm3):", round(volume, 2))

        resultados.append({
            "Paciente": paciente,
            "Mascara": tipo,
            "Geometria": geometria,
            "Binaria": binaria,
            "Voxels": voxels,
            "Volume_cm3": volume
        })

        if geometria and binaria:
            mascaras[tipo] = arr == 1

    if (
        "gtv_ref" in mascaras
        and "gtv_dicomseg" in mascaras
    ):

        a = mascaras["gtv_ref"]
        b = mascaras["gtv_dicomseg"]

        intersecao = np.count_nonzero(a & b)

        dice = (
            2 * intersecao
            / (np.count_nonzero(a) + np.count_nonzero(b))
        )

        print("Dice:", round(dice, 4))

        resultados.append({
            "Paciente": paciente,
            "Mascara": "Comparacao",
            "Dice": dice
        })

df = pd.DataFrame(resultados)

df.to_csv(
    OUTPUT / "mask_validation.csv",
    index=False
)

print("\nValidacao terminada")
