import importlib.util
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(name):
    path = Path(__file__).resolve().parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load("evaluate")
preparation = load("prepare_data")


class MetricTests(unittest.TestCase):
    def test_average_precision_groups_tied_scores(self):
        labels = np.array([True, False, True, False])
        self.assertEqual(evaluation.average_precision(labels, np.ones(4)), 0.5)
        self.assertEqual(evaluation.average_precision(labels, np.array([4, 1, 3, 2])), 1.0)
        self.assertAlmostEqual(evaluation.average_precision(labels, np.array([2, 4, 1, 3])), (1 / 3 + 2 / 4) / 2)

    def test_high_quantile_and_strict_threshold(self):
        threshold = evaluation.calibrated_threshold(np.arange(180))
        self.assertEqual(threshold, 171)
        self.assertEqual(int((np.arange(180) > threshold).sum()), 8)
        with self.assertRaises(ValueError):
            evaluation.calibrated_threshold(np.arange(200))

    def test_wilson_zero_and_full_have_nonzero_uncertainty(self):
        zero = evaluation.wilson(0, 100)
        full = evaluation.wilson(100, 100)
        self.assertAlmostEqual(zero["high"], 0.0369934982, places=8)
        self.assertAlmostEqual(full["low"], 0.9630065018, places=8)
        self.assertEqual(zero["value"], 0)
        self.assertEqual(full["value"], 1)

    def test_paired_bootstrap_preserves_identical_predictions(self):
        values = np.array([1, 0, 1, 0, 0, 1])
        self.assertEqual(evaluation.bootstrap_difference(values, values), {"value": 0.0, "low": 0.0, "high": 0.0})

    def test_invalid_metric_inputs_fail(self):
        with self.assertRaises(ValueError):
            evaluation.average_precision(np.array([1]), np.array([np.nan]))
        with self.assertRaises(ValueError):
            evaluation.wilson(0, 0)

    def test_pixel_metric_excludes_actual_padding_and_invalid_patches(self):
        canonical = {"transform": {"originalWidth": 224, "originalHeight": 200,
                    "resizedWidth": 224, "resizedHeight": 200,
                    "padLeft": 0, "padTop": 12, "inputSize": 224}}
        patches = np.ones(256, dtype=bool)
        patches[16] = False
        target, valid = evaluation.pixel_ground_truth({"maskPath": None}, canonical, {"validPatches": patches.tolist()})
        self.assertEqual(int(target.sum()), 0)
        self.assertFalse(valid[:12].any())
        self.assertFalse(valid[212:].any())
        self.assertFalse(valid[14:28, :14].any())
        self.assertEqual(int(valid.sum()), 224 * 200 - 14 * 14)


class DatasetSafetyTests(unittest.TestCase):
    def test_archive_path_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["../outside.jpg", "/outside.jpg", "candle/../../outside.jpg", "C:/outside.jpg", "candle\\outside.jpg"]:
                with self.assertRaises(ValueError):
                    preparation.safe_member_path(name, root)
            self.assertEqual(preparation.safe_member_path("candle/0001.JPG", root), (root / "candle/0001.JPG").resolve())

    def test_public_id_does_not_contain_label(self):
        identifier = evaluation.sample_id("candle-anomaly-001")
        self.assertNotIn("anomaly", identifier)
        self.assertEqual(identifier, evaluation.sample_id("candle-anomaly-001"))


class SplitLeakageTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"image": "fit.JPG", "split": "train", "label": "normal"},
            {"image": "cal.JPG", "split": "train", "label": "normal"},
            {"image": "test-good.JPG", "split": "test", "label": "normal"},
            {"image": "test-bad.JPG", "split": "test", "label": "anomaly"},
        ]
        names = ["fit", "calibration", "test", "test"]
        self.manifest = {"exclusions": [], "entries": [
            {"id": str(i), "officialImagePath": row["image"], "officialSplit": row["split"],
             "split": names[i], "label": row["label"], "sha256": str(i) * 64}
            for i, row in enumerate(self.rows)]}

    def validate(self):
        preparation.validate_manifest(self.manifest, self.rows, allowed_exclusions=set())

    def test_valid_disjoint_fixture_is_accepted(self):
        self.validate()

    def test_duplicate_ids_across_splits_are_rejected(self):
        self.manifest["entries"][2]["id"] = self.manifest["entries"][0]["id"]
        with self.assertRaisesRegex(ValueError, "IDs"):
            self.validate()

    def test_identical_source_hashes_across_splits_are_rejected(self):
        self.manifest["entries"][2]["sha256"] = self.manifest["entries"][0]["sha256"]
        with self.assertRaisesRegex(ValueError, "hashes"):
            self.validate()

    def test_fit_and_calibration_anomalies_are_rejected(self):
        clean = copy.deepcopy(self.manifest)
        for index in [0, 1]:
            self.manifest = copy.deepcopy(clean)
            self.manifest["entries"][index]["label"] = "anomaly"
            with self.assertRaisesRegex(ValueError, "normal images only"):
                self.validate()

    def test_swapping_test_normal_into_fit_is_rejected(self):
        self.manifest["entries"][0]["split"], self.manifest["entries"][2]["split"] = "test", "fit"
        with self.assertRaisesRegex(ValueError, "test membership"):
            self.validate()

    def test_dropping_test_image_is_rejected(self):
        self.manifest["entries"].pop()
        with self.assertRaisesRegex(ValueError, "source membership"):
            self.validate()

    def test_relabeling_test_anomaly_is_rejected(self):
        self.manifest["entries"][3]["label"] = "normal"
        with self.assertRaisesRegex(ValueError, "labels"):
            self.validate()

    def test_source_image_repeated_with_new_id_is_rejected(self):
        duplicate = copy.deepcopy(self.manifest["entries"][0])
        duplicate["id"] = "renamed-duplicate"
        duplicate["split"] = "calibration"
        self.manifest["entries"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "source image"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
