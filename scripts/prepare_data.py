"""Acquire the real VisA candle subset and freeze its evaluation split.

Only members explicitly listed in the pinned official split are copied from the
stream. No tar paths are passed to extract/extractall. The 1.93 GB tar is never
saved: candle is its first category, so the connection closes after 1,200 files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import shutil
import tarfile
import urllib.request

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "2a692ab575001cbde74d402d897a7286086c6199"
ARCHIVE = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
SPLIT_URL = f"https://raw.githubusercontent.com/amazon-science/spot-diff/{COMMIT}/split_csv/1cls.csv"
LICENSE_URL = f"https://raw.githubusercontent.com/amazon-science/spot-diff/{COMMIT}/LICENSE-DATASET"
EXPECTED_ETAG = '"05c830591a1172938cb714895c9e0cfb-113"'
SEED = 42
EXPECTED_SPLIT_SHA256 = "a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995"
EXCLUDED_IMAGE = "candle/Data/Images/Normal/0470.JPG"
SUPERSEDED_MANIFEST_SHA256 = "5782acc9ef6b196bb40d15cacf13a1dbc72f8f7de9bc5b9f816dacf286b28119"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def safe_member_path(name: str, destination: Path) -> Path:
    part = PurePosixPath(name)
    if part.is_absolute() or ".." in part.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive member: {name!r}")
    target = (destination / Path(*part.parts)).resolve()
    if not target.is_relative_to(destination.resolve()):
        raise ValueError(f"Archive member outside destination: {name!r}")
    return target


def download_small(url: str, target: Path) -> bytes:
    if target.exists():
        return target.read_bytes()
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return data


def acquire_members(names: set[str], raw: Path) -> None:
    missing = {name for name in names if not safe_member_path(name, raw).is_file()}
    if not missing:
        print("All required real source images/masks already exist.", flush=True)
        return
    print(f"Streaming {len(missing)} missing members from the official archive...", flush=True)
    with urllib.request.urlopen(ARCHIVE, timeout=120) as response:
        if response.headers.get("ETag") != EXPECTED_ETAG:
            raise RuntimeError("Official archive ETag changed; inspect before changing the pin.")
        with tarfile.open(fileobj=response, mode="r|", bufsize=1024 * 1024) as archive:
            count = 0
            for member in archive:
                if member.name not in missing:
                    continue
                if not member.isfile() or member.size > 30_000_000:
                    raise ValueError(f"Unexpected member type/size: {member.name}")
                output = safe_member_path(member.name, raw)
                output.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Cannot read member: {member.name}")
                temporary = output.with_suffix(output.suffix + ".partial")
                with temporary.open("wb") as destination:
                    shutil.copyfileobj(stream, destination)
                if temporary.stat().st_size != member.size:
                    raise IOError(f"Incomplete member: {member.name}")
                temporary.replace(output)
                missing.remove(member.name)
                count += 1
                if count % 100 == 0:
                    print(f"Saved {count} real files; {len(missing)} remaining.", flush=True)
                if not missing:
                    break
    if missing:
        raise RuntimeError(f"Archive missing {len(missing)} required files")


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def validate_manifest(manifest: dict, official_rows: list[dict], *,
                      expected_counts: dict | None = None,
                      allowed_exclusions: set[str] | None = None) -> None:
    """Reject leakage and changes to official held-out membership before scoring."""
    entries = manifest["entries"]
    ids = [entry["id"] for entry in entries]
    paths = [entry["officialImagePath"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs: fitting/calibration/test IDs must be disjoint")
    if len(paths) != len(set(paths)):
        raise ValueError("A source image appears more than once across manifest entries")
    official = {row["image"]: row for row in official_rows}
    if len(official) != len(official_rows):
        raise ValueError("Official source split contains duplicate image paths")
    exclusions = {entry["image"] for entry in manifest.get("exclusions", [])}
    if allowed_exclusions is not None and exclusions != allowed_exclusions:
        raise ValueError("Manifest exclusions differ from the audited permitted exclusions")
    if any(path not in official or official[path]["split"] != "train" for path in exclusions):
        raise ValueError("An exclusion may only remove an audited official training image")
    if set(paths) != set(official) - exclusions:
        raise ValueError("Manifest source membership differs from the official source minus audited exclusions")
    expected_test = {row["image"] for row in official_rows if row["split"] == "test"}
    actual_test = {entry["officialImagePath"] for entry in entries if entry["split"] == "test"}
    if actual_test != expected_test:
        raise ValueError("Official test membership changed; held-out test images must remain untouched")
    hash_splits: dict[str, set[str]] = {}
    counts = {name: 0 for name in ["fit", "calibration", "test"]}
    for entry in entries:
        name = entry["split"]
        if name not in counts:
            raise ValueError(f"Unknown split {name!r}")
        counts[name] += 1
        source = official[entry["officialImagePath"]]
        if name in {"fit", "calibration"} and entry["label"] != "normal":
            raise ValueError("Fitting and calibration must contain normal images only")
        if entry["label"] != source["label"] or entry["officialSplit"] != source["split"]:
            raise ValueError("Manifest labels or source split tags differ from official metadata")
        if name in {"fit", "calibration"} and source["split"] != "train":
            raise ValueError("Official test image leaked into fitting or calibration")
        digest = entry["sha256"]
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("Invalid source-image SHA256")
        hash_splits.setdefault(digest, set()).add(name)
    if any(len(splits) > 1 for splits in hash_splits.values()):
        raise ValueError("Identical source-image hashes occur across fitting/calibration/test splits")
    if expected_counts is not None and counts != expected_counts:
        raise ValueError(f"Unexpected split counts: {counts}")
    if manifest.get("counts", counts) != counts:
        raise ValueError("Manifest count metadata differs from actual entries")


def make_manifest(rows: list[dict[str, str]], raw: Path, split_data: bytes) -> dict:
    train = sorted(row["image"] for row in rows if row["split"] == "train")
    if len(train) != 900:
        raise ValueError(f"Expected 900 official candle training images, got {len(train)}")
    random.Random(SEED).shuffle(train)
    calibration = set(train[:180])
    replacement = None
    if EXCLUDED_IMAGE in calibration:
        calibration.remove(EXCLUDED_IMAGE)
        replacement = min(path for path in train if path not in calibration and path != EXCLUDED_IMAGE)
        calibration.add(replacement)
    entries = []
    for row in sorted(rows, key=lambda r: r["image"]):
        if row["image"] == EXCLUDED_IMAGE:
            continue
        path = safe_member_path(row["image"], raw)
        mask = safe_member_path(row["mask"], raw) if row["mask"] else None
        label = row["label"]
        split = "test" if row["split"] == "test" else "calibration" if row["image"] in calibration else "fit"
        with Image.open(path) as im:
            im.verify()
        if mask:
            with Image.open(mask) as im:
                im.verify()
        entries.append({
            "id": f"candle-{label}-{path.stem}", "split": split, "label": label,
            "imagePath": relative(path), "maskPath": relative(mask) if mask else None,
            "sha256": sha256(path.read_bytes()),
            "maskSha256": sha256(mask.read_bytes()) if mask else None,
            "officialSplit": row["split"], "officialImagePath": row["image"],
        })
    counts = {split: sum(e["split"] == split for e in entries) for split in ["fit", "calibration", "test"]}
    if counts != {"fit": 719, "calibration": 180, "test": 200}:
        raise ValueError(f"Unexpected counts: {counts}")
    return {
        "schemaVersion": 1, "category": "candle", "seed": SEED,
        "benchmarkVersion": "visa-candle-v2-audited-exclusion",
        "exclusions": [{"image": EXCLUDED_IMAGE, "sha256": sha256(safe_member_path(EXCLUDED_IMAGE, raw).read_bytes()),
                        "relatedOfficialTestImage": "candle/Data/Images/Normal/0469.JPG",
                        "reason": "Pre-inference visual audit confirmed essentially identical four-candle scene, marks and framing; conservative removal prevents this near-duplicate training image entering calibration or reference bank.",
                        "grayscale64Rmse": 1.1784192323684692, "grayscale64MaxDifference": 6.0,
                        "calibrationReplacement": replacement,
                        "replacementRule": "Lexicographically first original fitting image; no model scores were used."}],
        "supersededManifestSha256": SUPERSEDED_MANIFEST_SHA256,
        "source": {"url": ARCHIVE, "archiveEtag": EXPECTED_ETAG,
                   "repositoryCommit": COMMIT, "splitUrl": SPLIT_URL,
                   "splitSha256": sha256(split_data), "license": "CC-BY-4.0"},
        "counts": counts,
        "preprocessing": {"owner": "shared TypeScript kernel", "imageSize": 224,
                          "description": "Full image letterboxed to 224 square; exact resampling and padding are versioned in model artifact."},
        "entries": entries,
    }


def duplicate_audit(manifest: dict) -> dict:
    encoded: dict[str, list[dict]] = {}
    pixels: dict[str, list[dict]] = {}
    hashes = []
    for entry in manifest["entries"]:
        encoded.setdefault(entry["sha256"], []).append(entry)
        with Image.open(ROOT / entry["imagePath"]) as original:
            image = original.convert("RGB")
            digest = sha256(str(image.size).encode() + image.tobytes())
            pixels.setdefault(digest, []).append(entry)
            small = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
            bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
            dhash = sum(int(bit) << i for i, bit in enumerate(bits))
            thumbnail = np.asarray(image.convert("L").resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float32)
            hashes.append((entry, dhash, thumbnail))
    def groups(collection: dict) -> list:
        return [{"ids": [e["id"] for e in group], "splits": sorted({e["split"] for e in group})}
                for group in collection.values() if len(group) > 1]
    near = []
    coarse_candidates = 0
    for i, (first, ahash, first_thumbnail) in enumerate(hashes):
        for second, bhash, second_thumbnail in hashes[i + 1:]:
            if first["split"] != second["split"]:
                distance = (ahash ^ bhash).bit_count()
                if distance <= 2:
                    coarse_candidates += 1
                    difference = first_thumbnail - second_thumbnail
                    rmse = float(np.sqrt(np.mean(difference * difference)))
                    maximum = float(np.abs(difference).max())
                    if rmse <= 2.0 and maximum <= 12.0:
                        near.append({"first": first["id"], "second": second["id"], "hammingDistance": distance,
                                     "grayscale64Rmse": rmse, "grayscale64MaxDifference": maximum})
    exact = groups(encoded)
    pixel = groups(pixels)
    cross = [g for g in pixel if len(g["splits"]) > 1]
    return {
        "encodedDuplicateGroups": exact, "decodedPixelDuplicateGroups": pixel,
        "crossSplitExactDuplicateGroups": cross,
        "crossSplitPerceptualCandidates": near,
        "coarseDhashCandidates": coarse_candidates,
        "perceptualScreen": "64-bit grayscale dHash Hamming <= 2, confirmed by 64x64 grayscale RMSE <= 2 and maximum pixel difference <= 12 on a 0-255 scale. Fixed thresholds are a conservative screening heuristic, not proof of independence.",
        "limitations": "No manufacturing batch identities are provided. Perceptual screening cannot certify sample independence. Foundation-model pretraining overlap is unknown.",
    }


def prepare(data_dir: Path, apply_audited_exclusion: bool = False) -> None:
    raw = data_dir / "raw"
    metadata = data_dir / "source"
    split_data = download_small(SPLIT_URL, metadata / "1cls.csv")
    if sha256(split_data) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Pinned official split checksum mismatch")
    download_small(LICENSE_URL, metadata / "LICENSE-DATASET")
    rows = [row for row in csv.DictReader(io.StringIO(split_data.decode())) if row["object"] == "candle"]
    if len(rows) != 1100:
        raise ValueError(f"Expected 1,100 candle rows; got {len(rows)}")
    names = {row[key] for row in rows for key in ["image", "mask"] if row[key]}
    acquire_members(names, raw)
    manifest = make_manifest(rows, raw, split_data)
    validate_manifest(manifest, rows, expected_counts={"fit": 719, "calibration": 180, "test": 200},
                      allowed_exclusions={EXCLUDED_IMAGE})
    audit = duplicate_audit(manifest)
    audit["auditedExclusions"] = manifest["exclusions"]
    (data_dir / "duplicate-audit.json").write_bytes(json_bytes(audit))
    if audit["crossSplitExactDuplicateGroups"]:
        raise RuntimeError("Cross-split exact duplicate images found; inspect duplicate-audit.json before freezing.")
    target = data_dir / "split.json"
    output = json_bytes(manifest)
    if target.exists() and target.read_bytes() != output:
        if not apply_audited_exclusion or sha256(target.read_bytes()) != SUPERSEDED_MANIFEST_SHA256:
            raise RuntimeError("Frozen split differs. Refusing to overwrite; only the explicitly audited initial manifest exclusion may be applied.")
        (data_dir / "split-v1-superseded.json").write_bytes(target.read_bytes())
    target.write_bytes(output)
    (data_dir / "split.sha256").write_text(sha256(output) + "\n")
    print(f"Frozen {target}: {manifest['counts']}; {len(audit['crossSplitPerceptualCandidates'])} near-duplicate candidates.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/visa/candle")
    parser.add_argument("--apply-audited-exclusion", action="store_true", help="Replace only the exact initial v1 manifest with the documented pre-inference exclusion.")
    args = parser.parse_args()
    prepare(args.data_dir.resolve(), args.apply_audited_exclusion)
