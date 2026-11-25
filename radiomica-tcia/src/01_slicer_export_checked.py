# -*- coding: utf-8 -*-
"""Conservative Lung1 CT + primary-tumour export. Run INSIDE 3D Slicer.

Outputs are CANDIDATES requiring human approval. Never silently overwrites files.
For CT volumes with parent transforms (especially irregular-slice corrections),
this script deliberately blocks export rather than risk mismatched geometry.
"""
import os
import csv
import re
import traceback
from pathlib import Path

import numpy as np
import slicer
import vtk
from DICOMLib import DICOMUtils

RAW = Path(r"C:\Users\user\data_projeto\nsclc_radiomics")
OUT = Path(r"C:\Users\user\radiomica-tcia\data\nrrd_checked")
LOG = OUT / "review_candidates.csv"
# Optional: limit to one patient for testing, e.g. "LUNG1-030"; None = all
#ONLY_PATIENT = "LUNG1-030"

# Only exact, deliberately curated names. Not generic 'tumour' substring matching.
NAMES = {
    "rtstruct": {"gtv-1": 100, "gtv1": 95, "gtv_1": 95},
    "seg": {"neoplasm, primary": 80, "gtv-1": 100, "gtv1": 95},
}
FIELDS = ["patient_id", "status", "review", "source", "segment_name",
          "ct_series_uid", "structure_series_uid", "ct_referenced",
          "ct_frame_uid", "structure_frame_uid", "voxels", "volume_ml",
          "ct_shape_kji", "mask_shape_kji", "geometry_ok", "ct_parent_transform",
          "seg_parent_transform", "other_candidate_count", "other_candidates",
          "ct_path", "mask_path", "message"]


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().casefold())


def dicom_tag(db, files, tag):
    try:
        return (db.fileValue(files[0], tag) or "").strip()
    except Exception:
        return ""


def get_series(db, patient):
    result = []
    for study in db.studiesForPatient(patient):
        for uid in db.seriesForStudy(study):
            files = db.filesForSeries(uid)
            if files:
                result.append(dict(uid=uid, files=files,
                                   modality=dicom_tag(db, files, "0008,0060").upper(),
                                   frame=dicom_tag(db, files, "0020,0052")))
    return result


def referenced_ct_uids(file):
    """Read explicit referenced CT series UIDs, if present."""
    try:
        import pydicom
        ds = pydicom.dcmread(file, stop_before_pixels=True)
        found = set()
        for frame in getattr(ds, "ReferencedFrameOfReferenceSequence", []):
            for study in getattr(frame, "RTReferencedStudySequence", []):
                for series in getattr(study, "RTReferencedSeriesSequence", []):
                    found.add(str(series.SeriesInstanceUID))
        for series in getattr(ds, "ReferencedSeriesSequence", []):
            found.add(str(series.SeriesInstanceUID))
        return found
    except Exception as e:
        print("[WARN] Could not inspect DICOM references:", e)
        return set()


def load_nodes(uid, cls):
    result = []
    for node_id in DICOMUtils.loadSeriesByUID([uid]) or []:
        node = slicer.mrmlScene.GetNodeByID(node_id)
        if node and node.IsA(cls):
            result.append(node)
    return result


def matrix_equal(a, b, atol=1e-3):
    ma, mb = vtk.vtkMatrix4x4(), vtk.vtkMatrix4x4()
    a.GetIJKToRASMatrix(ma)
    b.GetIJKToRASMatrix(mb)
    aa = np.array([[ma.GetElement(i, j) for j in range(4)] for i in range(4)])
    bb = np.array([[mb.GetElement(i, j) for j in range(4)] for i in range(4)])
    return bool(np.allclose(aa, bb, atol=atol, rtol=0))


def choose_candidate(db, series):
    """Inspect every structure series and every segment, preserving source info."""
    found, all_names = [], []
    for s in series:
        kind = {"RTSTRUCT": "rtstruct", "SEG": "seg"}.get(s["modality"])
        if not kind:
            continue
        try:
            for node in load_nodes(s["uid"], "vtkMRMLSegmentationNode"):
                seg = node.GetSegmentation()
                for i in range(seg.GetNumberOfSegments()):
                    sid = seg.GetNthSegmentID(i)
                    name = seg.GetNthSegment(i).GetName() or ""
                    all_names.append("%s:%s" % (kind, name))
                    score = NAMES[kind].get(norm(name))
                    if score is not None:
                        found.append(dict(score=score, kind=kind, series=s,
                                          node=node, sid=sid, name=name,
                                          refs=referenced_ct_uids(s["files"][0])))
        except Exception as e:
            print("[WARN] Failed to load %s %s: %s" % (kind, s["uid"], e))
    found.sort(key=lambda c: c["score"], reverse=True)
    return found, all_names


def select_ct(series, candidate):
    cts = [s for s in series if s["modality"] == "CT"]
    refs = candidate["refs"]
    matched = [ct for ct in cts if ct["uid"] in refs]
    if len(matched) == 1:
        return matched[0], "yes"
    if len(matched) > 1:
        return None, "ambiguous_reference"
    if refs:
        return None, "reference_does_not_match_ct"
    if len(cts) == 1:
        return cts[0], "not_available_single_ct"
    return None, "ambiguous_ct"


def make_labelmap(seg_node, sid, ct_node):
    label = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "candidate_mask")
    ids = vtk.vtkStringArray()
    ids.InsertNextValue(sid)
    success = slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
        seg_node, ids, label, ct_node)
    if success is False:
        raise RuntimeError("ExportSegmentsToLabelmapNode returned False")
    return label


def process(db, pid, series):
    row = {k: "" for k in FIELDS}
    row.update(patient_id=pid, review="NOT_APPROVED")
    candidates, names = choose_candidate(db, series)
    row["other_candidates"] = " | ".join(names)
    row["other_candidate_count"] = len(candidates)
    if not candidates:
        row.update(status="NO_PRIMARY_TUMOUR_CANDIDATE",
                   message="Available segments: " + " | ".join(names))
        return row

    # Don't silently select among equally-ranked candidates from different SEG series.
    best = candidates[0]
    if len(candidates) > 1 and candidates[1]["score"] == best["score"]:
        row.update(status="AMBIGUOUS_CANDIDATES",
                   message="Multiple equal-ranked primary tumour segments; review series manually")
        return row

    row.update(source=best["kind"], segment_name=best["name"],
               structure_series_uid=best["series"]["uid"],
               structure_frame_uid=best["series"]["frame"])
    ct_series, ref_status = select_ct(series, best)
    row["ct_referenced"] = ref_status
    if ct_series is None:
        row.update(status="CT_REFERENCE_UNRESOLVED", message=ref_status)
        return row
    row.update(ct_series_uid=ct_series["uid"], ct_frame_uid=ct_series["frame"])
    if best["series"]["frame"] and ct_series["frame"] and best["series"]["frame"] != ct_series["frame"]:
        row.update(status="FRAME_OF_REFERENCE_MISMATCH",
                   message="CT and structure have different FrameOfReferenceUID")
        return row

    volumes = load_nodes(ct_series["uid"], "vtkMRMLScalarVolumeNode")
    if len(volumes) != 1:
        row.update(status="CT_LOAD_FAILED", message="Loaded CT volume count: %d" % len(volumes))
        return row
    ct = volumes[0]
    seg = best["node"]
    row["ct_parent_transform"] = ct.GetTransformNodeID() or ""
    row["seg_parent_transform"] = seg.GetTransformNodeID() or ""
    # Critical: log shows non-linear acquisition transforms for several patients.
    # Never flatten/harden those automatically or pretend exported NRRDs are aligned.
    if ct.GetParentTransformNode() or seg.GetParentTransformNode():
        row.update(status="TRANSFORM_REVIEW_REQUIRED",
                   message="CT/segmentation has a parent transform; inspect irregular slice geometry in Slicer")
        return row

    label = make_labelmap(seg, best["sid"], ct)
    try:
        a = slicer.util.arrayFromVolume(label)
        image = slicer.util.arrayFromVolume(ct)
        row["ct_shape_kji"] = str(tuple(image.shape))
        row["mask_shape_kji"] = str(tuple(a.shape))
        if a.shape != image.shape or not matrix_equal(label, ct):
            row.update(status="GEOMETRY_MISMATCH", geometry_ok="False",
                       message="Mask shape or full IJK-to-RAS matrix differs from CT")
            return row
        row["geometry_ok"] = "True"
        a[:] = (a > 0).astype(a.dtype)
        slicer.util.arrayFromVolumeModified(label)
        nvox = int(np.count_nonzero(a))
        row["voxels"] = nvox
        if nvox == 0:
            row.update(status="EMPTY_MASK", message="No tumour voxels")
            return row
        row["volume_ml"] = round(nvox * np.prod(ct.GetSpacing()) / 1000.0, 3)
        if not np.isfinite(image).all():
            row.update(status="INVALID_CT_VALUES", message="CT contains NaN or infinity")
            return row

        stem = pid + "_" + best["kind"] + "_candidate"
        ct_path = OUT / (stem + "_ct.nrrd")
        mask_path = OUT / (stem + "_mask.nrrd")
        row["ct_path"], row["mask_path"] = str(ct_path), str(mask_path)
        if ct_path.exists() or mask_path.exists():
            row.update(status="ALREADY_EXISTS", message="Not overwriting candidate files")
            return row
        if not slicer.util.saveNode(ct, str(ct_path)):
            raise RuntimeError("Could not save CT")
        if not slicer.util.saveNode(label, str(mask_path)):
            try:
                ct_path.unlink()
            except OSError:
                pass
            raise RuntimeError("Could not save mask")
        row.update(status="EXPORTED_NEEDS_VISUAL_REVIEW",
                   message="Inspect CT and mask in Slicer; verify primary tumour before approval")
        return row
    finally:
        slicer.mrmlScene.RemoveNode(label)


def write_log(rows):
    with LOG.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    if not RAW.is_dir():
        raise FileNotFoundError("DICOM folder does not exist: " + str(RAW))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with DICOMUtils.TemporaryDICOMDatabase() as db:
        print("[radiomica] Importing", RAW)
        DICOMUtils.importDicom(str(RAW), db)
        entries = []
        for patient in db.patients():
            series = get_series(db, patient)
            if series:
                pid = dicom_tag(db, series[0]["files"], "0010,0020") or str(patient)
               # if ONLY_PATIENT is None or ONLY_PATIENT == pid:
                   # entries.append((pid, series))
        print("[radiomica] Patients to inspect:", len(entries))
        for pid, series in sorted(entries):
            slicer.mrmlScene.Clear(0)
            slicer.app.processEvents()
            try:
                row = process(db, pid, series)
            except Exception as exc:
                traceback.print_exc()
                row = {k: "" for k in FIELDS}
                row.update(patient_id=pid, status="ERROR", review="NOT_APPROVED", message=repr(exc))
            rows.append(row)
            write_log(rows)
            print("[radiomica]", pid, row["status"], row["message"])
        slicer.mrmlScene.Clear(0)
    print("[radiomica] Review CSV:", LOG)
    print("[radiomica] No case is approved automatically")


if __name__ == "__main__":
    main()
