# Validation record

Recorded on 20 September 2026, Australia/Sydney, for experiment `candle-q8-wasm-v1-2d23b31d872d`. This record distinguishes measured behaviour from implemented behaviour and untested conditions. It is not a manufacturing acceptance or safety certification.

## Reproducible checks

The current local suites passed **37 TypeScript tests and 16 Python tests**. The [CI workflow](../.github/workflows/checks.yml) runs those suites and the production TypeScript/Vite build on Ubuntu with Node 24 and Python 3.12. Full dataset preparation, model inference and browser checks are separate from CI.

The tests exercise worker cancellation, rejection of concurrent jobs, stale messages, a mocked 180-second timeout, session timestamp merging, version compatibility, model input/patch contracts, reference scoring, calibration, corrupted artifact rejection and CSV escaping. Python tests cover metric calculations, padding exclusion, archive path safety, duplicate IDs/source paths, cross-split source hashes, normal-only fitting/calibration and unchanged official test membership.

Twelve further API tests inject missing/denied IndexedDB, quota failures during writes or clearing, a late transaction abort, inaccessible CacheStorage, a cache-write quota failure and a blocked artifact fetch. They verify errors reach callers, inaccessible caching falls back to hash-verified network bytes, and independent manual-review storage can still retain and update a note after a blocked inference download. These are mocked API boundaries: they do not prove the warning banners or recovery experience under an actual browser permission denial.

The [quick-start model helper](../scripts/fetch_model.mjs) was checked with a fresh download into an isolated directory: the 24.5 MB model from the immutable public release passed SHA-256 verification. The existing-file verification path also passed. This checks model restoration without repeating the full benchmark.

## Numerical parity and timing

The [saved browser result](../public/data/browser-validation.json) compares three frozen public images across DINOv2, classical and raw-pixel methods: **nine checks passed with zero observed image-score or patch-distance error**. The permitted tolerances were 0.002 for image scores and 0.005 for patch distances. The offline evaluator and browser both used pinned ONNX Runtime Web 1.22.0, single-threaded WASM.

The measured environment was Codex's embedded Chromium 153 on Windows x64. The development machine reports an Intel Core i7-12650H, 16 logical processors and 16,782,184,448 bytes of physical memory (15.63 GiB). The hardware values were read from the operating system; they do not measure application memory use.

Three warm DINOv2 processing measurements were 613.5, 618.9 and 647.6 ms. The first measured end-to-end request took 1,228 ms with existing browser caches. It is **not** a fresh-network startup measurement. The browser memory field was unavailable; peak process memory was not measured.

## Checks on the public demo

The following interactions were observed on the [public Hugging Face Space](https://huggingface.co/spaces/Daibolical/inspection-desk), using the browser above:

| Check | Observed result |
|---|---|
| Real inference | The full ten-image starter DINOv2 batch completed. |
| Space-page embedding | DINOv2 inference also completed inside the public Hugging Face Space iframe: the first image displayed score 0.674, threshold 0.673 and 1.07 seconds processing. |
| Decisions and persistence | All ten test review dispositions were set to **Uncertain** and remained after refresh. Saved inference results and the entered note were recovered. These were UI test decisions, not expert assessments of the products. |
| JSON export | A browser download was opened and parsed: ten review rows, eleven inference results (ten DINOv2 and one classical), with the saved note matching. |
| CSV export | A browser download was opened and parsed: ten data rows with the saved note matching. |
| Cancellation and retry | Immediate cancellation of a pixel-method batch restored manual review. Retrying one image completed, displaying score 2.965, threshold 4.239 and approximately 0.40 seconds processing. |
| Keyboard navigation | Right Arrow moved from the first to the second photo. |
| Image navigation | Zoom reached 125%; Fit returned the image to 100%. |
| Narrow viewport | At 390 × 844 pixels, a screenshot showed a scrollable queue and the full image area. This is a responsive-layout check, not a physical-phone performance test. |
| Console | No browser console errors were observed during these checks. |

## Explicitly unmeasured or untested

- Standalone Chrome and Edge were not separately tested; neither were Safari, Firefox or physical mobile devices.
- Fresh-cache network download time, slow-network behaviour, peak process memory and constrained-memory devices were not measured.
- Real-browser network blocking, denied IndexedDB access, quota exhaustion and private-browsing storage loss were not injected. The API-level failure tests above do not replace those end-to-end scenarios.
- The full 200-image benchmark was not rerun inside the browser. Browser parity uses three images; the full benchmark used the same WASM provider in Node.
- No manufacturing reviewer study, measured time saving, new-camera holdout or factory validation was performed.

The [case study](case-study.md) reports the fixed benchmark outcome: DINOv2 improves recall over the selected classical baseline but still misses 50 of 100 anomalies and raises seven false alarms versus two. The combined recall-gain/no-extra-false-alarms criterion was not met.

## Recorded walkthrough

The [two-minute walkthrough](https://daibolical-inspection-desk.static.hf.space/walkthrough.html) is a silent, captioned 120-second sequence assembled from timestamped screenshots of the actual public app at two frames per second, with held frames during capture gaps. Its captions retain the measured 50% recall and 7% false-alarm rate. Model files were already cached; this is not a cold-start measurement. The public player loaded, displayed its two-minute duration and played without an observed error. Local video metadata reported 120 seconds and a complete decode of all 240 frames passed. The [recording notes](walkthrough-script.md) describe its scope. The player and media are preserved in [Space commit `cbe1aa3896c6b539caac8a7ff3c86b198a7c1b06`](https://huggingface.co/spaces/Daibolical/inspection-desk/tree/cbe1aa3896c6b539caac8a7ff3c86b198a7c1b06).
