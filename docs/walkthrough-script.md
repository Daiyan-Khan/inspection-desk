# Two-minute walkthrough

This is the narration/caption script for the recorded demonstration. Record actual app behaviour; do not present cached screenshots as live inference or hide model-loading time.

| Time | On screen | Caption / narration |
|---|---|---|
| 0:00–0:15 | Queue and starter batch | Inspection Desk is a browser-based assistant for reviewing manufacturing photos. This version uses public candle images and keeps your decisions on your device. |
| 0:15–0:35 | Start the batch; loading/progress | A frozen visual model compares image patches with normal reference images. Processing happens locally in a worker, so the review interface remains usable. |
| 0:35–0:55 | Original image, overlay slider and zoom | The coloured overlay marks unusual regions. It is a suggestion, not a defect probability or a release decision. |
| 0:55–1:15 | Make a decision, type a note, move to another image | The reviewer can mark defect, acceptable variation or uncertain. The app saves decisions immediately and keeps them separate from dataset labels. |
| 1:15–1:35 | Switch to classical method and show evaluation | The learned features are compared against a classical colour-and-gradient method using the same data split and calibration target. The report shows actual test false alarms as well as recall. |
| 1:35–1:50 | Failure collection and limitations | The case study includes mistakes. Benchmark results do not establish factory accuracy or financial savings. |
| 1:50–2:00 | Export, reload/recovered decisions, repository link | Reviews can be exported, and the repository contains the preparation, evaluation and deployment instructions. |

Use the generated report's actual numbers if narrating performance. No result should be added to this script before evaluation completes.
