---
title: Inspection Desk
emoji: 🔎
colorFrom: green
colorTo: gray
sdk: static
app_build_command: npm ci && npm run build
app_file: dist/index.html
pinned: false
license: mit
---

# Inspection Desk

A browser-based inspection assistant with local review history. Review real manufacturing photos, compare pretrained visual features with classical computer vision, and record **defect**, **acceptable variation** or **uncertain** decisions.

The project asks a measurable question: do frozen DINOv2 features outperform a classical appearance comparator on the same public candle-inspection benchmark? The answer comes from the generated evaluation report, not from the application design.

## Run the app

Requires Node.js 22 or 24 and Python 3.12 for data/evaluation. No paid API or inference server is used.

The repository's npm configuration skips optional native CUDA downloads. The application and benchmark use WASM; no GPU installation is required.

```sh
npm ci
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
python scripts/prepare_data.py
npm run artifacts
python scripts/evaluate.py
npm run dev
```

Preparation downloads only the required members of the original VisA archive, checks source integrity, and preserves originals under ignored `data/`. The initial setup needs network access. Generated model weights are deliberately not stored in Git; the artifact builder downloads the pinned version and verifies its SHA-256.

Open the URL printed by Vite. Start with the ten-image collection, run analysis, adjust the anomaly overlay, record decisions and export JSON or CSV. Keys **1/2/3** choose a disposition; **J/K** or the arrow keys move between images when an input is not focused.

## What is implemented

- React + TypeScript review queue, zoomable image workspace and measured evaluation dashboard.
- Live CPU/WASM inference in a single Web Worker, with cancellation and retry.
- Pinned DINOv2-small q8 weights, classical multiscale Lab/gradient descriptors and a raw-pixel reference method.
- The same canonical images, TypeScript preprocessing/scoring and ONNX weights in the browser and offline evaluation.
- Versioned results and independent reviewer records stored in IndexedDB.
- Asset checksums, content-addressed caching and usable manual review if inference fails.
- Reproducible split audit, fixed calibration, confidence intervals, localisation metrics and error examples.

## Evaluation protocol

Use only VisA's candle category. The original official split contains 900 training normals and a 200-image test set (100 normal, 100 anomalous). A confirmed near-identical train/test scene was found during the pre-model audit. Training image `0470.JPG` was excluded; the official test set remains intact. The corrected partition contains **719 fit / 180 calibration / 200 test** images. The source paths, evidence and replacement rule appear in the generated duplicate audit.

All references and feature statistics come from fit normals only. Each method uses the same 1,024 reference patch locations. Thresholds use a finite-sample upper quantile on calibration normals at a 5% target false-alert rate. The calibration lock is written before test inference. No test labels select thresholds, display scales or methods.

Report recall and actual test false alarms together. Confidence intervals and paired recall differences describe this small benchmark; they do not establish factory performance. The 50% test anomaly prevalence is not a manufacturing defect rate. Heatmaps are coarse anomaly suggestions, not probabilities or causal explanations.

## Verify

```sh
npm test
python -m unittest discover -s scripts/tests
npm run build
```

For actual browser/offline WASM parity and browser timing, run the dev server and open `/validation.html`. Click **Run browser parity and timing checks**. It uses three frozen public samples across all three methods, accepts at most 0.002 absolute image-score error and 0.005 patch-distance error, and reports actual processing times. These are numerical agreement tolerances, not accuracy thresholds.

The reproducible tests cover cancellation, stale worker messages, timeouts, CSV safety, feature/scoring contracts, artifact integrity and data/metric invariants. Manual browser verification covers the actual review flow and persistence.

## Deployment

The target is a free public Hugging Face Static Space. Build locally, prepare the static upload with `npm run package:space`, and upload the contents of `.publish/space/`. This package contains the built site, model, reference banks, canonical samples, evaluation and licence notices. It requires no secret or paid hardware. See [deployment instructions](docs/deployment.md).

The GitHub repository contains application code, reproducible preparation/evaluation, frozen public reports and curated samples. Raw data, dependency folders and local caches are ignored. The prior product-discovery research is not included in the public application repository.

## Portfolio evidence

- [Architecture and boundaries](docs/architecture.md)
- [Case study](docs/case-study.md)
- [Walkthrough script](docs/walkthrough-script.md)
- [Third-party attribution](THIRD_PARTY_NOTICES.md)

No customer interviews, manufacturer adoption, time savings or commercial demand are claimed. No image uploads, shared accounts, automatic product release, defect severity or safety certification are provided.

Application code is MIT licensed. VisA images are CC BY 4.0; DINOv2 weights are Apache 2.0. Their terms remain separate.
