# Human Action Recognition (HAR) — Lightweight Experiments

**Author:** Shahriar Islam
**Project:** Human Action Recognition experiments on the KTH dataset using lightweight models (EfficientNet-B0 + Temporal GRU).
**Purpose:** Demonstrate iterative experimentation on temporal video classification, show reasoning, and provide a reproducible baseline.

---

## Summary

This repository contains a Jupyter notebook and supporting utility modules exploring multiple approaches to classify actions in video, with a strong emphasis on *how the modeling strategy evolves*:

1. **Ghost frame (averaging all frames into a single image)** — baseline / intuition test.
2. **Middle-frame heuristic** — single-frame baseline using the middle frame.
3. **Multi-frame averaging** — per-frame classification with temporal probability averaging.
4. **EfficientNet-B0 (grayscale) + Bi-GRU** — final hybrid model that explicitly learns temporal dynamics from per-frame embeddings.

The objective is not to achieve state-of-the-art results, but to document the *evolution of thought* from static frame heuristics to a clean, reproducible temporal baseline that balances model size, interpretability, and practicality.

---

## Repository Structure

```
.
├── HAR_Human_Action_Recognition.ipynb   # Main notebook: experiments, reflections, and full training logic
├── utils/
│   ├── data_setup.py                   # VideoDataset and data loading utilities
│   ├── engine.py                       # Training and evaluation loops
│   └── viz.py                          # Visualization helpers (plots, grids, confusion matrices)
├── README.md
└── LICENSE
```

> **Note:** Utility modules are generated programmatically at runtime using `%%writefile` within the notebook. This allows users to reproduce the full project by downloading and running the notebook alone, without requiring a pre-populated module structure.

Model definitions, checkpoint saving, and loading logic are implemented directly inside the notebook in the current version of the project.

---

## Quickstart

### 1) Environment

Create a virtual environment and install the required dependencies:

```bash
conda create -n har python=3.10 -y
conda activate har
pip install -r requirements.txt
```

Core dependencies include: `torch`, `torchvision`, `opencv-python`, `matplotlib`, `tqdm`, `torchmetrics`, `mlxtend`.

---

### 2) Dataset

Download the **KTH Human Action Dataset** manually from:

[https://www.csc.kth.se/cvap/actions/](https://www.csc.kth.se/cvap/actions/)

Place the dataset under the following structure:

```
Data/KTH/train/
Data/KTH/test/
```

Each action class should be in its own directory:
`walking`, `jogging`, `running`, `boxing`, `handwaving`, `handclapping`.

The notebook also contains helper cells that automate directory creation and basic dataset preparation.

---

### 3) Run the Notebook

Launch Jupyter and execute the notebook top-to-bottom:

```bash
jupyter notebook HAR_Human_Action_Recognition.ipynb
```

Key notebook sections:

* Data preprocessing and `VideoDataset` implementation
* Single-frame baselines (ghost frame, middle frame)
* Multi-frame averaging experiments
* EfficientNet-B0 + Bi-GRU temporal modeling
* Evaluation, confusion matrices, and qualitative visualizations

---

## Reproducing the Final Model (EfficientNet + GRU)

1. Prepare the dataset as described above.
2. Ensure paths inside the notebook match your local `Data/` directory.
3. Run the final model training section (EfficientNet-B0 + Bi-GRU).
4. Save model checkpoints as `.pth` files directly from the notebook.

Example checkpoint saving pattern:

```python
torch.save({
    'backbone': backbone.state_dict(),
    'temporal': temporal_model.state_dict(),
    'optimizer': optimizer.state_dict()
}, 'checkpoint.pth')
```

The notebook also includes helper functions such as `predict_video` and `predict_video_with_frame_probs` for per-video inference and analysis.

---

## Results & Observations (Summary)

* **Ghost frame averaging** produces blurry representations and weak performance.
* **Middle-frame heuristics** stabilize predictions but fail for motion-defined classes.
* **Multi-frame averaging** reduces noise but does not truly model motion.
* **EfficientNet + Bi-GRU** captures temporal structure and produces the most meaningful predictions.

Observed failure modes:

* Frames without visible subjects can lead to background memorization.
* Strong confusion between `walking`, `jogging`, and `running`, highlighting the need for richer temporal cues.

---

## Possible Extensions

* Integrate lightweight person detection to filter empty or irrelevant frames.
* Incorporate optical flow or pose-based features (e.g., MediaPipe).
* Explore compact 3D CNNs (R(2+1)D, I3D) or transformer-based video models.
* Add structured experiment tracking (TensorBoard or Weights & Biases).

---

## License & Attribution

This project is released under the **MIT License**.

Implementation and experiments by **Shahriar Islam**.
Design and modularization inspired in part by educational patterns from [learnpytorch.io](https://www.learnpytorch.io/).

---

## Contact

For questions or collaboration:

* Email: **[islam_shahriar@hotmail.com](mailto:islam_shahriar@hotmail.com)**
* LinkedIn: [https://www.linkedin.com/in/shahriar-islam-75567a160/](https://www.linkedin.com/in/shahriar-islam-75567a160/)
