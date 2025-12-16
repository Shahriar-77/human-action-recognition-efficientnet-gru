# Human Action Recognition (HAR) — Lightweight Experiments

**Author:** Shahriar Islam  
**Project:** Human Action Recognition experiments on KTH dataset using lightweight models (EfficientNet-B0 + Temporal GRU).  
**Purpose:** Demonstrate iterative experimentation on temporal video classification, show reasoning, and provide a reproducible baseline.

---

## Summary

This repository contains a Jupyter notebook and modular Python scripts exploring multiple approaches to classify actions in video:

1. **Ghost frame (averaging all frames into a single image)** — baseline / intuition test.
2. **Middle-frame heuristic** — single-frame baseline using the middle frame.
3. **Multi-frame averaging** — per-frame classification + temporal probability averaging.
4. **EfficientNet-B0 (grayscale) + Bi-GRU** — final hybrid model that learns temporal dynamics from per-frame embeddings.

The goal was not to build SOTA models, but to document the *evolution of thought* and reach a clean, reproducible temporal baseline that trades model size for interpretability and practicality.

---

## Repo Structure

```

.
├── HAR_Human_Action_Recognition.ipynb   # Main notebook with experiments + reflections
├── models/
│   ├── backbone.py                      # EfficientNetBackbone (grayscale)
│   ├── temporal.py                      # TemporalGRU and other temporal heads
├── data/
│   └── (scripted data setup helpers)    # utilities for downloading / preparing KTH
├── utils/
│   ├── data_setup.py                       # VideoDataset class
│   ├── viz.py                           # visualization helpers (plotting, grid)
│   └── engine.py                          # model trainer 
└── README.md

```

> Note: The above is a recommended structure. The notebook contains the working code; the `.py` files are modular extracts for easier reuse.

---

## Quickstart

### 1) Environment
Create a conda or virtualenv environment and install requirements:

```bash
conda create -n har python=3.10 -y
conda activate har
pip install -r requirements.txt
# required: torch, torchvision, opencv-python, matplotlib, tqdm, torchmetrics, mlxtend
````

### 2) Data

Download the KTH dataset manually from:
[https://www.csc.kth.se/cvap/actions/](https://www.csc.kth.se/cvap/actions/)

Or run the notebook cell that automates file placement (see the notebook for details). Place the dataset under:

```
Data/KTH/train
Data/KTH/test
```

Each class should be in its folder (walking, jogging, running, boxing, handwaving, handclapping).

### 3) Run the Notebook

Open the notebook and run cells in order:

```bash
jupyter notebook HAR_Human_Action_Recognition.ipynb
```

Key sections:

* Data preprocessing & `VideoDataset` implementation
* Model 1 (middle-frame) experiment
* Model 2 (multi-frame averaging)
* Model 3 (EfficientNet-B0 + Bi-GRU)
* Evaluation, confusion matrix, and visualizations

---

## How to reproduce the final model (EfficientNet + GRU)

1. Prepare the dataset as specified.
2. Edit any paths in the notebook to match your local `Data/` folder.
3. Train the final model (Model 3) via the notebook or run a script that loads the `EfficientNetBackbone` and `TemporalGRU` and runs `train_model(...)`.
4. Save checkpoints as `.pth` files:

   * Save `backbone.state_dict()` and `temporal_model.state_dict()` separately or together depending on preference.
   * Example:

     ```python
     torch.save({
         'backbone': backbone.state_dict(),
         'temporal': temporal_model.state_dict(),
         'optimizer': optimizer.state_dict()
     }, 'checkpoint.pth')
     ```

> Note: The notebook contains training loops, `tqdm`-based progress, plotting utilities, and `predict_video` / `predict_video_with_frame_probs` helpers for per-video inference.

---

## Results & Observations (summary)

* **Ghost averaging** produced blurry images and poor performance.
* **Middle-frame** heuristic stabilized some predictions but struggled with actions where motion defines the class (walking vs jogging).
* **Multi-frame averaging** reduced false positives from stray frames but did not learn motion patterns.
* **EfficientNet + Bi-GRU** learned temporal patterns and produced the most meaningful predictions. Overfitting was observed, likely due to dataset size vs model capacity; regularization and augmentation help.

Common failure modes:

* Frames without a visible subject are sometimes fed to the model (causing scene memorization).
* Confusions mainly among walking/jogging/running — these actions are best separated with strong temporal features or pose-based/optical-flow inputs.

---

## Next steps (suggested)

* Integrate lightweight person detection (YOLOv8n / MobileNet-SSD) to filter frames without subjects.
* Experiment with optical flow or pose (MediaPipe) features to capture subtle motion.
* Try small 3D CNNs (R(2+1)D, I3D) or TimeSformer for end-to-end spatiotemporal modeling.
* Add robust experiment logging (Weights & Biases, TensorBoard) and an evaluation script for cross-validation.

---

## License & Attribution

This project code is released under the MIT License.
Credit: Implementation & experiments by Shahriar Islam.
Special thanks to [learnpytorch.io](https://www.learnpytorch.io/) for design and implementation inspiration.

---

## Contact

If you have questions or want to collaborate, feel free to open an issue or contact me at: **[[islam_shahriar@hotmail.com](mailto:islam_shahriar@hotmail.com)]**
LinkedIn: **[(https://www.linkedin.com/in/shahriar-islam-75567a160/)]**

