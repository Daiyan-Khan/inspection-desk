# Third-party notices

## VisA dataset

Images and anomaly masks are from **VisA (Visual Anomaly)**, released by Amazon Science with *SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation* (ECCV 2022).

- Source: https://github.com/amazon-science/spot-diff
- Dataset licence: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- Official archive: https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
- Modifications: selected candle-category images, renamed public sample identifiers, aspect-preserving resizing and padding for canonical inference inputs. The review sample manifest identifies display images; the separate sample provenance record maps opaque identifiers to original source paths and hashes (`public/data/sample-provenance.json` in source, `data/sample-provenance.json` in the packaged site). Human review decisions are application data, not original dataset labels.

The upstream repository's code licence is separate from the dataset licence. This application does not imply endorsement by Amazon or the dataset authors.

## DINOv2

- Original model: Meta AI, https://github.com/facebookresearch/dinov2 and https://huggingface.co/facebook/dinov2-small
- ONNX conversion: https://huggingface.co/Xenova/dinov2-small
- Model licence: Apache License 2.0, https://www.apache.org/licenses/LICENSE-2.0
- Exact conversion revision and file checksum are recorded in the artifact manifest. This project uses the base DINOv2-small model, not XRay-DINO.

## Application dependencies

Dependency versions are recorded in package-lock.json. React, Vite, ONNX Runtime, idb, lucide and their dependencies retain their own licences. Distribution must retain their applicable notices.
