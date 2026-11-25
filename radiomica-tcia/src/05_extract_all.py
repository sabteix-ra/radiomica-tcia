from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# Diretorios
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PARAMS = BASE / "config" / "params.yaml"
OUT = DATA / "features"

OUT.mkdir(parents=True, exist_ok=True)

# Configuracao do PyRadiomics
extractor = featureextractor.RadiomicsFeatureExtractor(
    str(PARAMS)
)

# Pacientes
patients = [
    "LUNG1-" + str(i).zfill(3)
    for i in range(10, 101, 10)
]

# Mascaras selecionadas para a analise principal
primary = {
    patient: "gtv_ref"
    for patient in patients
}

primary["LUNG1-030"] = "gtv_dicomseg"

# Resultados
results = []
comparisons = []
logs = []


def find_file(name):

    files = list(DATA.rglob(name))

    if len(files) != 1:
        raise ValueError(
            f"Esperado um ficheiro {name}; "
            f"encontrados {len(files)}: {files}"
        )

    return files[0]


def check_geometry(ct, mask):

    if ct.GetSize() != mask.GetSize():
        raise ValueError("Dimensoes diferentes")

    if not np.allclose(
        ct.GetSpacing(), mask.GetSpacing()
    ):
        raise ValueError("Spacing diferente")

    if not np.allclose(
        ct.GetOrigin(), mask.GetOrigin()
    ):
        raise ValueError("Origem diferente")

    if not np.allclose(
        ct.GetDirection(), mask.GetDirection()
    ):
        raise ValueError("Orientacao diferente")


# Extracao coletiva
for patient in patients:

    print("\nPaciente:", patient)

    masks = {}

    for mask_type in ["gtv_ref", "gtv_dicomseg"]:

        mask_name = f"{patient}_{mask_type}.nrrd"

        if not list(DATA.rglob(mask_name)):

            status = "NOT_AVAILABLE"

            if mask_type == primary[patient]:
                status = "ERROR_PRIMARY_MISSING"

            logs.append({
                "patient": patient,
                "mask": mask_type,
                "status": status
            })

            continue

        try:

            ct_path = find_file(
                f"{patient}_ct.nrrd"
            )

            mask_path = find_file(mask_name)

            print("Mascara:", mask_type)

            ct = sitk.ReadImage(str(ct_path))
            mask = sitk.ReadImage(str(mask_path))

            # Validacao geometrica
            check_geometry(ct, mask)

            array = sitk.GetArrayFromImage(mask)

            if not np.all(np.isin(array, [0, 1])):
                raise ValueError("Mascara nao binaria")

            binary = array == 1

            n_voxels = np.count_nonzero(binary)

            if n_voxels == 0:
                raise ValueError("Mascara vazia")

            volume_cm3 = (
                n_voxels *
                np.prod(ct.GetSpacing()) / 1000
            )

            masks[mask_type] = (
                binary, volume_cm3
            )

            # Extracao das caracteristicas
            features = extractor.execute(
                ct, mask
            )

            row = {
                "patient": patient,
                "mask": mask_type,
                "primary": mask_type == primary[patient],
                "n_voxels": n_voxels,
                "volume_cm3": volume_cm3
            }

            for key, value in features.items():

                if not key.startswith("diagnostics_"):
                    row[key] = float(value)

            results.append(row)

            logs.append({
                "patient": patient,
                "mask": mask_type,
                "status": "OK",
                "ct_path": str(ct_path),
                "mask_path": str(mask_path)
            })

            print("Volume (cm3):", volume_cm3)
            print("Extracao concluida")

        except Exception as error:

            print("ERRO:", error)

            logs.append({
                "patient": patient,
                "mask": mask_type,
                "status": "ERROR",
                "message": str(error)
            })

    # Comparacao das duas mascaras
    if "gtv_ref" in masks and "gtv_dicomseg" in masks:

        A, vol_A = masks["gtv_ref"]
        B, vol_B = masks["gtv_dicomseg"]

        intersection = np.count_nonzero(A & B)

        dice = (
            2 * intersection /
            (np.count_nonzero(A) + np.count_nonzero(B))
        )

        comparisons.append({
            "patient": patient,
            "dice": dice,
            "volume_ref_cm3": vol_A,
            "volume_seg_cm3": vol_B,
            "volume_difference_pct":
                100 * (vol_B - vol_A) / vol_A,
            "different_voxels":
                np.count_nonzero(A != B)
        })


# Guardar resultados
df = pd.DataFrame(results)

df.to_csv(
    OUT / "features_all.csv",
    index=False
)

if not df.empty:

    df_primary = df[df["primary"] == True]

    df_primary.to_csv(
        OUT / "features_primary.csv",
        index=False
    )

    print("\nMascaras processadas:", len(df))
    print("Pacientes na analise principal:", len(df_primary))

pd.DataFrame(comparisons).to_csv(
    OUT / "mask_comparison.csv",
    index=False
)

pd.DataFrame(logs).to_csv(
    OUT / "extraction_log.csv",
    index=False
)

print("\nProcessamento terminado")
print("Resultados guardados em:", OUT)