

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from torchmetrics.classification import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import torch.nn.functional as F


# Set deivce
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ✅ Define preprocessing (convert to PIL first)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

def load_video_frames(video_path, num_frames=16, frame_skip=2):
    """Returns sampled frames from a video as transformed tensors (PIL -> Tensor)."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        # Convert to grayscale (ETH/KTH)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = Image.fromarray(frame)  # ✅ Convert to PIL Image
        frame = transform(frame)        # ✅ Apply transforms safely
        frames.append(frame)
    cap.release()
    return torch.stack(frames)  # shape (T, 1, 224, 224)



def show_middle_frame_batch(dataloader, max_images=16):
    """
    Visualize the middle-frame inputs that are fed to the model.
    Assumes dataloader yields: X [B, T, C, H, W], y [B]
    """
    # Grab one batch
    X, y = next(iter(dataloader))

    B, T, C, H, W = X.shape
    mid_idx = T // 2

    # Get the middle frames
    mid_frames = X[:, mid_idx]   # [B, C, H, W]

    # Limit number of images
    num_show = min(B, max_images)

    cols = 4
    rows = (num_show + cols - 1) // cols

    plt.figure(figsize=(12, 3 * rows))

    for i in range(num_show):
        frame = mid_frames[i].squeeze(0).numpy()   # From [1,H,W] to [H,W]

        plt.subplot(rows, cols, i + 1)
        plt.imshow(frame, cmap="gray")
        plt.title(f"Label: {y[i].item()}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()




def plot_confusion_matrix_avg_over_frames(data_loader, model, train_dataset):

    """
    Computes and plots the confusion matrix by averaging per-frame predictions
    for each video. Suitable for models without explicit temporal heads.
    """

    # Ensure model is in eval mode
    model.eval()

    # Collect all predictions and labels
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for X, y in data_loader:
            # X shape: [B, T, C, H, W]
            B, T, C, H, W = X.shape

            # Flatten frames into batch for prediction
            X_flat = X.view(B * T, C, H, W).to(device)

            # Get frame-level logits
            frame_logits = model(X_flat)  # [B*T, num_classes]

            # Reshape back to [B, T, num_classes]
            frame_logits = frame_logits.view(B, T, -1)

            # Convert to probabilities
            frame_probs = F.softmax(frame_logits, dim=-1)

            # Average probabilities over frames
            avg_probs = frame_probs.mean(dim=1)  # [B, num_classes]

            # Predicted class per video
            preds = avg_probs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute confusion matrix
    num_classes = len(train_dataset.classes)
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    cm = confmat(all_preds, all_labels).numpy()

    # Plot using mlxtend
    fig, ax = plot_confusion_matrix(
        conf_mat=cm,
        class_names=train_dataset.classes,
        figsize=(8, 6),
        show_normed=True,
        colorbar=True
    )

    plt.title("Confusion Matrix on Test Set (Avg over frames)")
    plt.show()


def plot_confusion_matrix_temporal_GRU(train_dataset,backbone,temporal_model,data_loader):

    '''Plots a confusion matrix for the last model Effnet_backbone with temporal GRU'''





    backbone.eval()
    temporal_model.eval()
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch in data_loader:
            X = batch["frames"].to(device)  # [B, T, C, H, W]
            y = batch["labels"].to(device)  # [B]

            B, T, C, H, W = X.shape


            # Backbone → embeddings
            embeddings = backbone(X)
            logits = temporal_model(embeddings)  # [B, num_classes]

            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute confusion matrix
    num_classes = len(train_dataset.classes)
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    cm = confmat(all_preds, all_labels).numpy()

    # Plot
    fig, ax = plot_confusion_matrix(
        conf_mat=cm,
        class_names=train_dataset.classes,
        figsize=(8, 6),
        show_normed=True,
        colorbar=True
    )
    plt.title("Confusion Matrix (Test Set)")
    plt.show()

