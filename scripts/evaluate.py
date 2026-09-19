"""Evaluate frozen predictions, then export measured public results and samples.

No model fitting or test-driven threshold selection occurs here. Thresholds are
normal-calibration high quantiles. Test labels enter only after thresholds freeze.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import shutil

import numpy as np
from PIL import Image
from prepare_data import EXPECTED_SPLIT_SHA256, EXCLUDED_IMAGE, validate_manifest

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"dinov2": ("aiScore", "aiHeatmap", "aiMs", "Frozen DINOv2"),
           "classical": ("baselineScore", "baselineHeatmap", "baselineMs", "Colour + gradient baseline"),
           "pixel": ("pixelScore", "pixelHeatmap", "pixelMs", "Raw-pixel baseline")}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sample_id(internal: str) -> str:
    return "sample-" + sha256(internal.encode())[:12]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict:
    if total <= 0:
        raise ValueError("A binomial interval needs at least one observation")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"value": p, "low": max(0.0, center - half), "high": min(1.0, center + half)}


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Non-interpolated AP, aggregating tied scores before advancing recall."""
    labels = np.asarray(labels, dtype=np.bool_).ravel()
    scores = np.asarray(scores).ravel()
    if labels.size != scores.size or not np.isfinite(scores).all():
        raise ValueError("AP needs matching labels and finite scores")
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    tp = np.cumsum(labels[order], dtype=np.int64)
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), labels.size - 1]
    cumulative = tp[ends]
    delta = np.diff(np.r_[0, cumulative])
    return float(np.sum((cumulative / (ends + 1)) * delta) / positives)


def calibrated_threshold(scores: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.size != 180 or not np.isfinite(values).all():
        raise ValueError("Calibration requires exactly 180 finite normal scores")
    return float(np.quantile(values, 0.95, method="higher"))


def bootstrap_difference(a: np.ndarray, b: np.ndarray, repetitions: int = 5000) -> dict:
    """Paired image bootstrap for a difference in binary detection rates."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("Paired bootstrap requires matching nonempty observations")
    rng = np.random.default_rng(42)
    indices = rng.integers(0, a.size, size=(repetitions, a.size))
    difference = a - b
    distribution = difference[indices].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {"value": float(difference.mean()), "low": float(low), "high": float(high)}


def bootstrap_ap_difference(labels: np.ndarray, a: np.ndarray, b: np.ndarray, repetitions: int = 2000) -> dict:
    rng = np.random.default_rng(42)
    positive, negative = np.flatnonzero(labels), np.flatnonzero(~labels)
    values = []
    for _ in range(repetitions):
        ids = np.r_[rng.choice(positive, positive.size), rng.choice(negative, negative.size)]
        values.append(average_precision(labels[ids], a[ids]) - average_precision(labels[ids], b[ids]))
    low, high = np.quantile(values, [0.025, 0.975])
    return {"value": average_precision(labels, a) - average_precision(labels, b), "low": float(low), "high": float(high)}


def canonical_entries(value: object) -> list[dict]:
    if isinstance(value, list):
        return value
    return value.get("images", value.get("entries", []))


def transform_for(canonical: dict) -> dict:
    if "transform" in canonical:
        return canonical["transform"]
    box = canonical["letterbox"]
    return {"originalWidth": canonical["originalWidth"], "originalHeight": canonical["originalHeight"],
            "resizedWidth": box["width"], "resizedHeight": box["height"],
            "padLeft": box["x"], "padTop": box["y"], "inputSize": 224}


def pixel_ground_truth(entry: dict, canonical: dict, prediction: dict) -> tuple[np.ndarray, np.ndarray]:
    transform = transform_for(canonical)
    target = np.zeros((224, 224), dtype=np.bool_)
    if entry["maskPath"]:
        with Image.open(ROOT / entry["maskPath"]) as image:
            mask = np.asarray(image.convert("L").resize((transform["resizedWidth"], transform["resizedHeight"]), Image.Resampling.NEAREST)) > 0
        x, y = transform["padLeft"], transform["padTop"]
        target[y:y + mask.shape[0], x:x + mask.shape[1]] = mask
    valid_patches = prediction.get("validPatches", canonical.get("validPatches"))
    if valid_patches is None:
        raise ValueError("validPatches required for localisation evaluation")
    valid = np.repeat(np.repeat(np.asarray(valid_patches, dtype=np.bool_).reshape(16, 16), 14, axis=0), 14, axis=1)
    content = np.zeros((224, 224), dtype=np.bool_)
    x, y = transform["padLeft"], transform["padTop"]
    content[y:y + transform["resizedHeight"], x:x + transform["resizedWidth"]] = True
    valid &= content
    return target, valid


def evaluate(data_dir: Path, public_dir: Path) -> dict:
    manifest_bytes = (data_dir / "split.json").read_bytes()
    split = json.loads(manifest_bytes)
    official_bytes = (data_dir / "source/1cls.csv").read_bytes()
    if sha256(official_bytes) != EXPECTED_SPLIT_SHA256:
        raise ValueError("Pinned official split checksum mismatch before evaluation")
    official_rows = [row for row in csv.DictReader(io.StringIO(official_bytes.decode())) if row["object"] == "candle"]
    validate_manifest(split, official_rows, expected_counts={"fit": 719, "calibration": 180, "test": 200},
                      allowed_exclusions={EXCLUDED_IMAGE})
    predictions_bytes = (data_dir / "predictions.json").read_bytes()
    prediction_file = json.loads(predictions_bytes)
    predictions = {p["id"]: p for p in prediction_file["images"]}
    canonical = {p["id"]: p for p in canonical_entries(json.loads((data_dir / "canonical.json").read_text()))}
    entries = split["entries"]
    held_out_ids = {e["id"] for e in entries if e["split"] != "fit"}
    if len(predictions) != len(prediction_file["images"]) or set(predictions) != held_out_ids:
        raise ValueError("Predictions must contain every calibration/test image exactly once, and no fitting images")
    for entry in entries:
        if sha256((ROOT / entry["imagePath"]).read_bytes()) != entry["sha256"]:
            raise ValueError(f"Changed source image {entry['id']}")
        if entry["maskPath"] and sha256((ROOT / entry["maskPath"]).read_bytes()) != entry["maskSha256"]:
            raise ValueError(f"Changed source mask {entry['id']}")
    calibration = [e for e in entries if e["split"] == "calibration"]
    test = [e for e in entries if e["split"] == "test"]
    if len(calibration) != 180 or len(test) != 200 or any(e["label"] != "normal" for e in calibration):
        raise ValueError("Unexpected calibration/test membership")
    labels = np.array([e["label"] == "anomaly" for e in test])
    if labels.sum() != 100:
        raise ValueError("Test must contain 100 normal and 100 anomaly images")
    artifacts_path = public_dir / "artifacts/manifest.json"
    artifact = json.loads(artifacts_path.read_text())
    if artifact["splitVersion"] != sha256(manifest_bytes):
        raise ValueError("Artifact was not built from this frozen benchmark manifest")
    if artifact["version"] != prediction_file["modelVersion"]:
        raise ValueError("Prediction model version differs from public artifact version")
    methods, scores_by_method, flags_by_method = [], {}, {}
    pixel_labels, valid_masks = [], []
    for entry in test:
        truth, valid = pixel_ground_truth(entry, canonical[entry["id"]], predictions[entry["id"]])
        pixel_labels.append(truth[valid])
        valid_masks.append(valid)
    all_pixel_labels = np.concatenate(pixel_labels)
    for method, (score_key, heatmap_key, timing_key, title) in METHODS.items():
        if any(score_key not in predictions[e["id"]] for e in calibration + test):
            raise ValueError(f"Missing {method} scores")
        threshold = calibrated_threshold(np.array([predictions[e["id"]][score_key] for e in calibration]))
        if not math.isclose(artifact["methods"][method]["threshold"], threshold, rel_tol=1e-7, abs_tol=1e-10):
            raise ValueError(f"Artifact {method} threshold differs from independent calibration")
        scores = np.array([predictions[e["id"]][score_key] for e in test], dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite test scores")
        flagged = scores > threshold
        tp, fp = int((flagged & labels).sum()), int((flagged & ~labels).sum())
        pixel_scores = []
        for entry, valid in zip(test, valid_masks):
            heat = np.asarray(predictions[entry["id"]][heatmap_key], dtype=np.float32).reshape(16, 16)
            pixel_scores.append(np.repeat(np.repeat(heat, 14, axis=0), 14, axis=1)[valid])
        timings = np.array([predictions[e["id"]].get(timing_key, float("nan")) for e in test])
        if not np.isfinite(timings).all():
            raise ValueError(f"Measured {timing_key} latency required; refusing invented values")
        methods.append({"method": method, "label": title, "threshold": threshold,
                        "truePositives": tp, "falseNegatives": 100 - tp, "falsePositives": fp, "trueNegatives": 100 - fp,
                        "recall": wilson(tp, 100), "falseAlarmRate": wilson(fp, 100),
                        "averagePrecision": average_precision(labels, scores),
                        "pixelAveragePrecision": average_precision(all_pixel_labels, np.concatenate(pixel_scores)),
                        "medianLatencyMs": float(np.median(timings)), "p95LatencyMs": float(np.quantile(timings, 0.95))})
        scores_by_method[method], flags_by_method[method] = scores, flagged
    recall_difference = bootstrap_difference(flags_by_method["dinov2"][labels], flags_by_method["classical"][labels])
    fpr_difference = bootstrap_difference(flags_by_method["dinov2"][~labels], flags_by_method["classical"][~labels])
    ap_difference = bootstrap_ap_difference(labels, scores_by_method["dinov2"], scores_by_method["classical"])
    if recall_difference["low"] > 0:
        conclusion = "DINOv2 has higher observed defect recall than the classical baseline at independently calibrated thresholds; assess the actual false-alarm rates alongside this difference."
    elif recall_difference["high"] < 0:
        conclusion = "The classical baseline has higher observed defect recall at independently calibrated thresholds. This benchmark does not establish an AI advantage."
    else:
        conclusion = "The paired recall interval includes zero. This benchmark does not establish a reliable recall improvement from DINOv2 over the classical baseline."
    ai_metric = next(method for method in methods if method["method"] == "dinov2")
    classical_metric = next(method for method in methods if method["method"] == "classical")
    meaningful_recall_gain = 0.10
    fpr_no_greater = ai_metric["falseAlarmRate"]["value"] <= classical_metric["falseAlarmRate"]["value"]
    combined_criterion_met = recall_difference["value"] >= meaningful_recall_gain and fpr_no_greater
    criterion_status = "met on this benchmark" if combined_criterion_met else "not met"
    conclusion += (f" This is a measured tradeoff: DINOv2 flagged {ai_metric['falsePositives']} of 100 normal images"
                   f" versus {classical_metric['falsePositives']} for the classical baseline, and still missed"
                   f" {ai_metric['falseNegatives']} of 100 anomalies. The combined recall-gain/no-extra-false-alarms"
                   f" criterion is {criterion_status}. This is not evidence of a factory win. Review assistance only.")
    split_version = sha256(manifest_bytes)
    report = {
        "schemaVersion": 1, "status": "complete", "artifactVersion": prediction_file["modelVersion"],
        "splitVersion": split_version, "evaluatedAt": datetime.now(timezone.utc).isoformat(),
        "counts": {"fit": split["counts"]["fit"], "calibration": 180, "testNormal": 100, "testAnomaly": 100},
        "calibrationTarget": 0.05, "methods": methods, "pairedRecallDifference": recall_difference,
        "acceptanceCriteria": {"meaningfulRecallGain": meaningful_recall_gain,
                               "observedRecallGain": recall_difference["value"],
                               "meaningfulRecallGainMet": recall_difference["value"] >= meaningful_recall_gain,
                               "currentObservedFPRNoGreater": fpr_no_greater,
                               "combinedCriterionMet": combined_criterion_met,
                               "comparison": "DINOv2 versus classical at the already frozen, independently calibrated thresholds; reporting only, no threshold retuning."},
        "pairedFalseAlarmDifference": fpr_difference, "pairedAveragePrecisionDifference": ap_difference,
        "bootstrap": {"pairedRateRepetitions": 5000, "pairedStratifiedAPRepetitions": 2000, "seed": 42},
        "conclusion": conclusion,
        "limitations": [
            "One benchmark category and 100 anomalies; factory accuracy and defect prevalence are unknown.",
            "Intervals describe image sampling uncertainty, not new-factory or new-camera performance.",
            "One visually confirmed near-duplicate training photo was excluded before inference, leaving 719 fitting images; all 200 official test images remain. No batch identities are supplied, so screening cannot prove complete sample independence.",
            "Foundation-model pretraining overlap is unknown.",
            "Thresholds target 5% calibration false alarms; test false alarms can differ.",
            "Pixel AP uses raw 16x16 patch scores expanded to 224 pixels, excluding padded patches; it is not native-resolution segmentation.",
            "Latency is offline CPU feature extraction and scoring; image decoding, first model load and browser execution are excluded. Browser timing must be measured separately.",
            "Review assistance only. An anomaly score is not a defect probability or a release decision.",
        ],
        "environment": prediction_file.get("environment", f"Offline extraction; {platform.system()} {platform.machine()}"),
        "provenance": {"manifestSha256": split_version, "predictionsSha256": sha256(predictions_bytes),
                       "repositoryCommit": split["source"]["repositoryCommit"], "sourceSplitSha256": split["source"]["splitSha256"]},
        "results": [{"sampleId": sample_id(e["id"]), "label": e["label"],
                     "scores": {m: float(scores_by_method[m][i]) for m in METHODS},
                     "flagged": {m: bool(flags_by_method[m][i]) for m in METHODS}} for i, e in enumerate(test)],
    }
    write_json(public_dir / "data/evaluation.json", report)
    write_json(data_dir / "evaluation.json", report)
    samples = export_samples(test, canonical, report, split, public_dir)
    internal_by_public = {sample_id(e["id"]): e["id"] for e in test}
    fixtures = []
    for sample in [s for s in samples if "starter" in s["collections"]][:3]:
        prediction = predictions[internal_by_public[sample["id"]]]
        for method, (score_key, heatmap_key, _, _) in METHODS.items():
            fixtures.append({"sampleId": sample["id"], "method": method,
                             "score": prediction[score_key], "heatmap": prediction[heatmap_key]})
    write_json(public_dir / "data/parity-fixtures.json", {"artifactVersion": report["artifactVersion"], "fixtures": fixtures})
    # Public reproducibility metadata is separate from the review-queue manifest.
    (public_dir / "data/benchmark-manifest.json").write_bytes(manifest_bytes)
    shutil.copyfile(data_dir / "duplicate-audit.json", public_dir / "data/duplicate-audit.json")
    shutil.copyfile(data_dir / "source/LICENSE-DATASET", public_dir / "data/LICENSE-DATASET.txt")
    return report


def export_samples(test: list[dict], canonical: dict, report: dict, split: dict, public_dir: Path) -> list[dict]:
    # Fixed IDs before looking at scores; every resulting success/failure remains visible.
    starter = sorted(test, key=lambda e: sha256(e["id"].encode()))[:10]
    by_public = {sample_id(e["id"]): e for e in test}
    findings = []
    for row in report["results"]:
        ai, baseline = row["flagged"]["dinov2"], row["flagged"]["classical"]
        error = ai != (row["label"] == "anomaly")
        if ai != baseline or error:
            findings.append(by_public[row["sampleId"]])
    findings = sorted(findings, key=lambda e: sha256(e["id"].encode()))[:10]
    selected = {e["id"]: e for e in starter + findings}
    destination = public_dir / "samples"
    destination.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, entry in enumerate(selected.values(), 1):
        identifier = sample_id(entry["id"])
        original_target = destination / f"{identifier}.jpg"
        input_target = destination / f"{identifier}-input.png"
        shutil.copyfile(ROOT / entry["imagePath"], original_target)
        canonical_path = Path(canonical[entry["id"]]["path"])
        if not canonical_path.is_absolute():
            canonical_path = ROOT / canonical_path
        shutil.copyfile(canonical_path, input_target)
        transform = transform_for(canonical[entry["id"]])
        samples.append({"id": identifier, "name": f"Inspection photo {index:02d}",
                        "imageUrl": f"/samples/{original_target.name}", "inputUrl": f"/samples/{input_target.name}",
                        "sourcePath": "VisA candle image; provenance in the separate evaluation record",
                        "width": transform["originalWidth"], "height": transform["originalHeight"],
                        "collections": (["starter"] if entry in starter else []) + (["findings"] if entry in findings else []),
                        "transform": transform})
    write_json(public_dir / "data/sample-manifest.json", {
        "schemaVersion": 1, "dataset": "VisA candle", "sourceUrl": "https://github.com/amazon-science/spot-diff",
        "license": "CC-BY-4.0", "splitVersion": report["splitVersion"], "samples": samples,
    })
    write_json(public_dir / "data/sample-provenance.json", {
        "attribution": "VisA dataset: Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, Onkar Dabeer (ECCV 2022). CC BY 4.0.",
        "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
        "sourceCommit": split["source"]["repositoryCommit"],
        "modifications": "Original JPEGs copied unchanged. Input PNGs resized and grey-letterboxed to 224x224 by the pinned shared preprocessing kernel.",
        "samples": [{"sampleId": sample_id(e["id"]), "sourcePath": e["officialImagePath"], "originalSha256": e["sha256"]} for e in selected.values()],
    })
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/visa/candle")
    parser.add_argument("--public-dir", type=Path, default=ROOT / "public")
    args = parser.parse_args()
    outcome = evaluate(args.data_dir, args.public_dir)
    print(json.dumps({"conclusion": outcome["conclusion"], "methods": outcome["methods"]}, indent=2))
