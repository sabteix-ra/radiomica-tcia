
from pathlib import Path
import pandas as pd
from radiomics import featureextractor

# Diretório principal do projeto
BASE = Path(r"C:\Users\user\radiomica-tcia")

# Ficheiros de entrada
DATA = BASE / "data" / "nrrd_checked"
PARAMS = BASE / "config" / "params.yaml"

CT = DATA / "LUNG1-030_seg_candidate_ct.nrrd"
MASK = DATA / "LUNG1-030_seg_candidate_mask.nrrd"

# Diretório de saída
OUTPUT = BASE / "data" / "features"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Verificar os ficheiros
for ficheiro in [CT, MASK, PARAMS]:
    if not ficheiro.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {ficheiro}")

# Inicializar PyRadiomics
extractor = featureextractor.RadiomicsFeatureExtractor(
    str(PARAMS)
)

# Extrair características
print("A extrair características radiómicas...")

resultados = extractor.execute(
    str(CT),
    str(MASK)
)

# Excluir informação de diagnóstico
features = {
    chave: float(valor)
    for chave, valor in resultados.items()
    if not chave.startswith("diagnostics_")
}

# Guardar resultados
df = pd.DataFrame([features])
df.insert(0, "Paciente", "LUNG1-030")

destino = OUTPUT / "LUNG1-030_features.csv"
df.to_csv(destino, index=False)

print("Extração concluída!")
print("Número de características:", len(features))
print("Resultados guardados em:", destino)

# Mostrar algumas características
for chave, valor in features.items():
    if chave.startswith("original_shape_") or chave.startswith("original_firstorder_"):
        print(chave, ":", round(valor, 3))
