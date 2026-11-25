# PyRadiomics — Caracterização quantitativa de tumores pulmonares

## Objetivo

Demonstrar a utilização da biblioteca PyRadiomics na extração de características quantitativas de imagens de tomografia computorizada (TC) de pacientes com carcinoma pulmonar de não pequenas células.

Projeto desenvolvido no âmbito da unidade curricular de Métodos de Imagem Médica, Engenharia Biomédica, NOVA FCT.

## Dados

Coleção NSCLC-Radiomics, disponibilizada pelo The Cancer Imaging Archive (TCIA).

https://www.cancerimagingarchive.net/collection/nsclc-radiomics/

Foram analisados 10 pacientes e 17 máscaras tumorais.

## Ferramentas

- Python 3.9
- 3D Slicer
- PyRadiomics
- SimpleITK
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Lifelines

## Metodologia

1. Importação de imagens DICOM.
2. Exportação das imagens e máscaras em formato NRRD.
3. Controlo de qualidade das segmentações.
4. Extração de características radiómicas.
5. Análise descritiva e concordância entre segmentações.
6. Integração dos dados clínicos.
7. Análise exploratória de sobrevivência.
8. Validação cruzada Leave-One-Out.

## Resultados principais

| Parâmetro | Resultado |
|---|---:|
| Pacientes | 10 |
| Máscaras tumorais | 17 |
| Características radiómicas por máscara | 1130 |
| Dice médio | 0,9714 |
| Mediana de sobrevivência | 457 dias |
| C-index aparente do volume tumoral | 0,721 |
| C-index LOOCV do volume tumoral | 0,395 |

## Instalação

Criar um ambiente virtual Python 3.9 e instalar as dependências:

`pip install -r requirements.txt`

Os dados médicos originais não estão incluídos neste repositório e devem ser obtidos através do TCIA.

## Limitações

A reduzida dimensão amostral e a ausência de validação externa impedem a demonstração de utilidade prognóstica clínica. Este projeto possui uma finalidade exclusivamente académica e demonstrativa.