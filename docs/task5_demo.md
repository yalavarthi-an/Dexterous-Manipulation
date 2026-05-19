# Task 5: Demo Video and Evaluation

## Overview

Task 5 asked for a **demo video** of the dexterous-grasp pipeline and an **evaluation** of success on the configured objects.

**Published demo (submission):** [YouTube — quad-view grasp demo](https://youtu.be/tW-jJHIFZbQ)

The recording shows each of the five YCB targets in sequence: two fixed tabletop views plus two lateral views rigged to the palm, matching the mosaic produced by `scripts/record_all_grasps.py`.

---

## How I produced the footage

Offline I run the same grasp proposals and executor as Tasks 3–4, but step MuJoCo with a fixed `--stride`, capture RGB from four virtual cameras each frame, and stitch a **2×2 H.264 MP4** via `imageio[ffmpeg]` (dependency is already listed in `requirements.txt`).

```bash
conda activate pathon
cd /path/to/pathon_takehome_test
python scripts/visualize_grasps.py --save
python scripts/record_all_grasps.py --out outputs/demo_all_grasps.mp4
```

Details, flags (`--objects`, `--stride`, `--no-top-flip`), and optional OpenCV captions are documented in the **`record_all_grasps.py` (Task 5 demo)** section of [`docs/tools.md`](tools.md).

`outputs/` is gitignored locally; reviewers who need a file copy can regenerate the MP4 with the commands above.

---

## Evaluation summary

Quantitative regression is summarized in **`docs/task4_execution.md`** and **`report/technical_report.md`**.

| Metric | Result |
|--------|--------|
| Successful grasps (of 5) | **4** (banana, cracker box, mustard bottle, tennis ball) |
| Failures | **1** (mug — `GRASP_SLIP`) |
| Success rate | **80%** (exceeds the brief’s minimum of ≥3 lifts) |

The video is the qualitative companion to those numbers: it shows nominal behavior where the scripted lift succeeds, and the mug segment illustrates the unreliable rim grasp I discuss in Task 4’s failure analysis.
