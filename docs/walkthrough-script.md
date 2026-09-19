# Two-minute recorded walkthrough

This is a **silent, captioned recording of the actual hosted app**, assembled from timestamped screenshots at two frames per second. Where capture gaps occur, the most recent screenshot is held until the next captured frame. The two-minute timeline is preserved; app states and inference outputs are not reconstructed.

The model assets were already cached from earlier deployment checks. This recording does **not** demonstrate an uncached first visit or establish cold-start performance.

| Time | Recorded action | Caption |
|---|---|---|
| 0:00–0:16 | Starter queue in the hosted app | Inspection Desk supports human review of manufacturing photos. This demo uses public candle images; review decisions stay in this browser. |
| 0:16–0:42 | Actual batch processing on the device | This batch is analysed on the device. Model files were already cached from earlier deployment checks, so this is not a cold-start measurement. |
| 0:42–1:03 | Anomaly overlay and its opacity control | The overlay highlights unusual appearance. It is a review suggestion, not a defect probability or a release decision. |
| 1:03–1:19 | Reviewer selects uncertain and writes a note | The reviewer records uncertain and adds a note. Decisions and notes are saved locally, separately from benchmark labels. |
| 1:19–1:34 | Classical method applied to the same image | The same image can be reviewed with a conventional colour-and-gradient baseline. Both methods use the same frozen data split and calibration target. |
| 1:34–1:59 | Frozen evaluation report and observed tradeoff | On the held-out test, DINOv2 reached 50% recall with 7% false alerts; the classical baseline reached 0% recall with 2% false alerts. The combined success criterion was not met. |
| 1:59–2:00 | Representative mistakes | Review assistance only — including its mistakes. |

The caption timings follow the recorded actions. No opacity percentage is claimed. The brief final caption can be read by pausing the player.

Deliverables: `walkthrough.webm` (VP8, 240 frames, 2 fps, 120 seconds), `walkthrough.html` (standard browser player), and `walkthrough.vtt` (optional caption track). Captions are also drawn in a separate band below the screenshots. The recording has no audio.
