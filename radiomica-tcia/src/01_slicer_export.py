# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
01_slicer_export.py -- exporta CT + GTV-1 para NRRD (NSCLC-Radiomics / Lung1)

Corre DENTRO do 3D Slicer (precisa da extensão SlicerRT para ler RTSTRUCT).
Para cada doente encontrado em RAW:
  * importa o DICOM para uma base de dados TEMPORÁRIA (não mexe na tua base do Slicer);
  * carrega a série de CT e a estrutura (RTSTRUCT ou DICOM SEG);
  * exporta o CT e a máscara binária do GTV-1, na grelha do CT, para NRRD;
  * regista tudo em export_log_<fonte>.csv (inclui parâmetros de aquisição).

Configuração por variáveis de ambiente (ou edita os valores por omissão):
  RADIOMICA_RAW       pasta com os DICOM (contém as pastas LUNG1-xxx)
  RADIOMICA_OUT       pasta de saída (data/nrrd)
  RADIOMICA_SOURCE    rtstruct (omissão) | seg
  RADIOMICA_SEGMENT   nome da estrutura (omissão: GTV-1)
  RADIOMICA_LIMIT     n.º máximo de doentes (0 = todos)
  RADIOMICA_HEADLESS  1 = arranca sozinho após o início do Slicer e fecha no fim

Ficheiros produzidos (por doente):
  <PatientID>_ct.nrrd              CT em HU
  <PatientID>_gtv_ref.nrrd         máscara 0/1 a partir da RTSTRUCT
  <PatientID>_gtv_dicomseg.nrrd    máscara 0/1 a partir do DICOM SEG (se SOURCE=seg)
"""
import csv
import os
import traceback

import numpy as np
import qt
import slicer
import vtk
from DICOMLib import DICOMUtils

# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------
os.environ['RADIOMICA_RAW'] = r'"C:\Users\user\data_projeto\nsclc_radiomics"'
HOME_PROJ = os.path.join(os.path.expanduser("~"), "radiomica-tcia")

RAW = r"C:\Users\user\data_projeto\nsclc_radiomics"

OUT = os.environ.get(
    "RADIOMICA_OUT",
    os.path.join(HOME_PROJ, "data", "nrrd")
)
SOURCE = os.environ.get("RADIOMICA_SOURCE", "rtstruct").lower()
SEGMENT = os.environ.get("RADIOMICA_SEGMENT", "GTV-1")
LIMIT = int(os.environ.get("RADIOMICA_LIMIT", "0"))
HEADLESS = os.environ.get("RADIOMICA_HEADLESS", "0") == "1"

if SOURCE not in ("rtstruct", "seg"):
    raise ValueError("RADIOMICA_SOURCE deve ser 'rtstruct' ou 'seg'")
STRUCT_MODALITY = {"rtstruct": "RTSTRUCT", "seg": "SEG"}[SOURCE]
MASK_SUFFIX = {"rtstruct": "gtv_ref", "seg": "gtv_dicomseg"}[SOURCE]

LOG_FIELDS = [
    "patient_id", "status", "message", "n_ct_series", "n_struct_series",
    "ct_series", "struct_series", "segment_name", "voxels_gtv",
    "dims", "spacing_xyz", "geom_ok",
    "slice_thickness", "kernel", "kvp", "manufacturer",
]


# ----------------------------------------------------------------------------
# Auxiliares sobre a base de dados DICOM
# ----------------------------------------------------------------------------
def tag(db, files, dicom_tag):
    """Lê uma etiqueta DICOM (ex.: '0008,0060') do primeiro ficheiro da série."""
    try:
        return (db.fileValue(files[0], dicom_tag) or "").strip()
    except Exception:
        return ""


def list_series(db, patient_uid):
    out = []
    for study in db.studiesForPatient(patient_uid):
        for series in db.seriesForStudy(study):
            files = db.filesForSeries(series)
            if not files:
                continue
            out.append({
                "uid": series,
                "files": files,
                "n": len(files),
                "modality": tag(db, files, "0008,0060").upper(),
            })
    return out


def referenced_series_uids(struct_file):
    """UIDs das séries de imagem referidas por uma RTSTRUCT ou DICOM SEG."""
    try:
        import pydicom
    except ImportError:
        slicer.util.pip_install("pydicom")
        import pydicom
    ds = pydicom.dcmread(struct_file, stop_before_pixels=True)
    uids = set()
    for fr in getattr(ds, "ReferencedFrameOfReferenceSequence", []):
        for st in getattr(fr, "RTReferencedStudySequence", []):
            for se in getattr(st, "RTReferencedSeriesSequence", []):
                uids.add(str(se.SeriesInstanceUID))
    for se in getattr(ds, "ReferencedSeriesSequence", []):
        uids.add(str(se.SeriesInstanceUID))
    return uids


def pick_ct(series, struct):
    """Escolhe a série de CT: a referida pela estrutura; senão, a maior."""
    cts = [s for s in series if s["modality"] == "CT"]
    if not cts:
        return None, 0
    if len(cts) == 1:
        return cts[0], 1
    try:
        ref = referenced_series_uids(struct["files"][0])
        match = [s for s in cts if s["uid"] in ref]
        if match:
            return match[0], len(cts)
    except Exception:
        traceback.print_exc()
    return max(cts, key=lambda s: s["n"]), len(cts)


# ----------------------------------------------------------------------------
# Carregamento e exportação
# ----------------------------------------------------------------------------
def load_volume(series_uid):
    for node_id in DICOMUtils.loadSeriesByUID([series_uid]):
        node = slicer.mrmlScene.GetNodeByID(node_id)
        if node is not None and node.IsA("vtkMRMLScalarVolumeNode"):
            return node
    return None


def load_segmentations(series_uid):
    nodes = []
    for node_id in DICOMUtils.loadSeriesByUID([series_uid]):
        node = slicer.mrmlScene.GetNodeByID(node_id)
        if node is not None and node.IsA("vtkMRMLSegmentationNode"):
            nodes.append(node)
    return nodes


def segment_names(seg_node):
    seg = seg_node.GetSegmentation()
    return [seg.GetNthSegment(i).GetName() for i in range(seg.GetNumberOfSegments())]


def find_segment(seg_node, name):
    """(id, nome) do segmento cujo nome coincide com `name` (sem distinguir maiúsculas)."""
    seg = seg_node.GetSegmentation()
    wanted = name.lower()
    exact, prefix = None, None
    for i in range(seg.GetNumberOfSegments()):
        sname = (seg.GetNthSegment(i).GetName() or "").strip()
        sid = seg.GetNthSegmentID(i)
        if sname.lower() == wanted and exact is None:
            exact = (sid, sname)
        elif sname.lower().startswith(wanted) and prefix is None:
            prefix = (sid, sname)
    return exact or prefix or (None, None)


def export_mask(seg_node, segment_id, ct_node, path):
    """Exporta um segmento para labelmap 0/1 na grelha do CT e grava em NRRD."""
    ids = vtk.vtkStringArray()
    ids.InsertNextValue(segment_id)
    lm = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "gtv_mask")
    ok = slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
        seg_node, ids, lm, ct_node)
    if ok is False:
        raise RuntimeError("ExportSegmentsToLabelmapNode devolveu False")

    arr = slicer.util.arrayFromVolume(lm)
    arr[:] = (arr > 0)                      # garante 0/1 (GTV-1 = 1)
    slicer.util.arrayFromVolumeModified(lm)
    n_vox = int(arr.sum())
    if n_vox == 0:
        raise RuntimeError("máscara vazia após exportação")

    dims_ok = tuple(lm.GetImageData().GetDimensions()) == tuple(ct_node.GetImageData().GetDimensions())
    spacing_ok = np.allclose(lm.GetSpacing(), ct_node.GetSpacing(), atol=1e-4)
    if not slicer.util.saveNode(lm, path):
        raise RuntimeError("falha ao gravar " + path)
    return n_vox, bool(dims_ok and spacing_ok)


# ----------------------------------------------------------------------------
# Um doente
# ----------------------------------------------------------------------------
def process_patient(db, pid, series):
    row = {k: "" for k in LOG_FIELDS}
    row["patient_id"] = pid
    ct_path = os.path.join(OUT, "%s_ct.nrrd" % pid)
    mask_path = os.path.join(OUT, "%s_%s.nrrd" % (pid, MASK_SUFFIX))

    structs = [s for s in series if s["modality"] == STRUCT_MODALITY]
    if not structs:
        row["status"] = "sem_estrutura"
        row["message"] = "sem série " + STRUCT_MODALITY
        return row
    struct = structs[0]
    row["n_struct_series"] = len(structs)

    ct, n_ct = pick_ct(series, struct)
    row["n_ct_series"] = n_ct
    if ct is None:
        row["status"] = "sem_ct"
        row["message"] = "sem série de CT"
        return row
    row["ct_series"] = ct["uid"]
    row["struct_series"] = struct["uid"]

    # Parâmetros de aquisição (para o README / critérios de inclusão)
    row["slice_thickness"] = tag(db, ct["files"], "0018,0050")
    row["kernel"] = tag(db, ct["files"], "0018,1210")
    row["kvp"] = tag(db, ct["files"], "0018,0060")
    row["manufacturer"] = tag(db, ct["files"], "0008,0070")

    if os.path.exists(ct_path) and os.path.exists(mask_path):
        row["status"] = "ja_existe"
        return row

    slicer.mrmlScene.Clear(0)
    slicer.app.processEvents()

    ct_node = load_volume(ct["uid"])
    slicer.app.processEvents()
    if ct_node is None:
        row["status"] = "ct_nao_carregou"
        row["message"] = "o CT não carregou como volume escalar"
        return row

    seg_nodes = load_segmentations(struct["uid"])
    slicer.app.processEvents()
    if not seg_nodes:
        row["status"] = "estrutura_nao_carregou"
        row["message"] = ("a %s não carregou como segmentação "
                          "(SlicerRT instalado? Slicer reiniciado?)" % STRUCT_MODALITY)
        return row

    found = None
    all_names = []
    for node in seg_nodes:
        all_names += segment_names(node)
        sid, sname = find_segment(node, SEGMENT)
        if sid:
            found = (node, sid, sname)
            break
    if found is None:
        row["status"] = "sem_gtv1"
        row["message"] = "estruturas: " + "; ".join(all_names)
        return row

    node, sid, sname = found
    row["segment_name"] = sname

    n_vox, geom_ok = export_mask(node, sid, ct_node, mask_path)
    row["voxels_gtv"] = n_vox
    row["geom_ok"] = geom_ok
    dims = ct_node.GetImageData().GetDimensions()
    row["dims"] = "x".join(str(d) for d in dims)
    row["spacing_xyz"] = "x".join("%.3f" % s for s in ct_node.GetSpacing())

    if not os.path.exists(ct_path):
        if not slicer.util.saveNode(ct_node, ct_path):
            raise RuntimeError("falha ao gravar " + ct_path)

    row["status"] = "ok" if geom_ok else "ok_verificar_geometria"
    return row


# ----------------------------------------------------------------------------
# Ciclo principal
# ----------------------------------------------------------------------------
def write_log(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    log_path = os.path.join(OUT, "export_log_%s.csv" % SOURCE)
    rows = []

    if not os.path.isdir(RAW):
        raise FileNotFoundError(
            "A pasta RADIOMICA_RAW não existe: %s\n"
            "Define-a antes de correr: os.environ['RADIOMICA_RAW'] = r'C:\\...\\data\\raw'" % RAW)
    n_files = sum(len(fs) for _, _, fs in os.walk(RAW))
    print("[radiomica] %d ficheiros encontrados em %s" % (n_files, RAW))
    if n_files == 0:
        raise RuntimeError("A pasta %s está vazia (sem ficheiros)." % RAW)

    with DICOMUtils.TemporaryDICOMDatabase() as db:
        print("[radiomica] a importar %s ..." % RAW)
        DICOMUtils.importDicom(RAW, db)
        slicer.app.processEvents()

        patient_uids = list(db.patients())
        print("[radiomica] doentes na base temporária: %d" % len(patient_uids))
        if not patient_uids:
            print("[radiomica] AVISO: o Slicer não reconheceu nenhum DICOM nessa pasta. "
                  "Confirma que os ficheiros são DICOM e testa a importação pelo módulo DICOM.")

        entries = []
        for puid in patient_uids:
            series = list_series(db, puid)
            if not series:
                print("[radiomica] AVISO: doente %s sem séries/ficheiros na base" % puid)
                continue
            pid = tag(db, series[0]["files"], "0010,0020") or str(puid)
            entries.append((pid, series))
        entries.sort(key=lambda e: e[0])
        if LIMIT > 0:
            entries = entries[:LIMIT]
        print("[radiomica] %d doentes a processar (fonte: %s)" % (len(entries), SOURCE))

        for i, (pid, series) in enumerate(entries, 1):
            print("[radiomica] (%d/%d) %s" % (i, len(entries), pid))
            try:
                row = process_patient(db, pid, series)
            except Exception as exc:
                traceback.print_exc()
                row = {k: "" for k in LOG_FIELDS}
                row["patient_id"] = pid
                row["status"] = "erro"
                row["message"] = repr(exc)
            print("[radiomica]     -> %s %s" % (row["status"], row["message"]))
            rows.append(row)
            write_log(log_path, rows)
            slicer.app.processEvents()

        slicer.mrmlScene.Clear(0)

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("[radiomica] concluído: %s" % counts)
    print("[radiomica] registo: %s" % log_path)


def _run():
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        if HEADLESS:
            slicer.app.quit()


if HEADLESS:
    # Adia a execução até o arranque do Slicer terminar (carregar DICOM durante
    # o arranque é instável em algumas versões).
    qt.QTimer.singleShot(1000, _run)
else:
    _run()