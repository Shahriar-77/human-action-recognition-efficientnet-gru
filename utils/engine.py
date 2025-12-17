
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models as M
from pathlib import Path
from glob import glob
import cv2
import random
from tqdm.auto import tqdm
import torch.nn.functional as F

# Seed helper
def set_seeds(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in dataloader:
        # Take middle frame [B, T, 1, H, W] → [B, 1, H, W]
        mid_idx = X.shape[1] // 2
        X = X[:, mid_idx, :, :, :].to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (y_pred.argmax(dim=1) == y).float().mean().item()

    return train_loss / len(dataloader), train_acc / len(dataloader)


def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            mid_idx = X.shape[1] // 2
            X = X[:, mid_idx, :, :, :].to(device)
            y = y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            test_loss += loss.item()
            test_acc += (y_pred.argmax(dim=1) == y).float().mean().item()

    return test_loss / len(dataloader), test_acc / len(dataloader)

def train_model(model, train_loader, test_loader, optimizer, loss_fn, epochs, device):
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_loader, loss_fn, device)

        print(f"Epoch {epoch+1} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


### For multi-frame per video  


def train_step_2(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss, total_acc = 0, 0

    for X, y in dataloader:
        # X: [B, T, C, H, W] — frames per video
        B, T, C, H, W = X.shape
        X = X.view(B * T, C, H, W).to(device)  # merge batch & time

        y = y.to(device)

        # Forward through CNN in batch
        frame_logits = model(X)               # [B*T, num_classes]
        frame_logits = frame_logits.view(B, T, -1)

        # ✅ Temporal softmax smoothing (per video)
        frame_probs = F.softmax(frame_logits, dim=-1)
        avg_probs = frame_probs.mean(dim=1)   # [B, num_classes]

        # Compute loss on averaged probs
        loss = loss_fn(avg_probs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += (avg_probs.argmax(1) == y).float().mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader)


def test_step_2(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            B, T, C, H, W = X.shape
            X = X.view(B * T, C, H, W).to(device)
            y = y.to(device)

            frame_logits = model(X)
            frame_logits = frame_logits.view(B, T, -1)

            frame_probs = F.softmax(frame_logits, dim=-1)
            avg_probs = frame_probs.mean(dim=1)

            loss = loss_fn(avg_probs, y)
            total_loss += loss.item()
            total_acc += (avg_probs.argmax(1) == y).float().mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader)


def train_2(model, train_loader, test_loader, optimizer, loss_fn, epochs, device):
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in tqdm(range(epochs)):
        tr_loss, tr_acc = train_step_2(model, train_loader, loss_fn, optimizer, device)
        te_loss, te_acc = test_step_2(model, test_loader, loss_fn, device)
        print(f"Epoch {epoch+1} | Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | "
              f"Test Loss: {te_loss:.4f}, Test Acc: {te_acc:.4f}")
        results["train_loss"].append(tr_loss)
        results["train_acc"].append(tr_acc)
        results["test_loss"].append(te_loss)
        results["test_acc"].append(te_acc)
    return results



## EFFICIENT_NET BACKBONE WITH GRU

# ---------------------------
# Train step 
# ---------------------------
def train_step_3(model, backbone, dataloader, optimizer, loss_fn, device):
    model.train()
    backbone.eval()  # backbone frozen
    total_loss, total_acc = 0, 0

    loop = tqdm(dataloader, desc="Training", leave=False)
    for batch in loop:
        X = batch["frames"].to(device)
        y = batch["labels"].to(device)

        with torch.no_grad():
            embeddings = backbone(X)  # [B, T, 1280]

        logits = model(embeddings)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += (logits.argmax(1) == y).float().mean().item()

        loop.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{(logits.argmax(1) == y).float().mean().item():.4f}"
        })

    return total_loss / len(dataloader), total_acc / len(dataloader)


# ---------------------------
# Test step 
# ---------------------------
def test_step_3(model, backbone, dataloader, loss_fn, device):
    model.eval()
    backbone.eval()
    total_loss, total_acc = 0, 0

    loop = tqdm(dataloader, desc="Testing ", leave=False)
    with torch.no_grad():
        for batch in loop:
            X = batch["frames"].to(device)
            y = batch["labels"].to(device)
            embeddings = backbone(X)
            logits = model(embeddings)
            loss = loss_fn(logits, y)

            total_loss += loss.item()
            total_acc += (logits.argmax(1) == y).float().mean().item()

            loop.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(logits.argmax(1) == y).float().mean().item():.4f}"
            })

    return total_loss / len(dataloader), total_acc / len(dataloader)


# ---------------------------
# Main training loop 
# ---------------------------
def train_model_3(backbone, temporal_model, train_loader, test_loader, optimizer, loss_fn, device, epochs=20):
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    epoch_loop = tqdm(range(epochs), desc="Epochs", unit="epoch")
    for epoch in epoch_loop:
        tr_loss, tr_acc = train_step_3(temporal_model, backbone, train_loader, optimizer, loss_fn, device)
        te_loss, te_acc = test_step_3(temporal_model, backbone, test_loader, loss_fn, device)

        results["train_loss"].append(tr_loss)
        results["train_acc"].append(tr_acc)
        results["test_loss"].append(te_loss)
        results["test_acc"].append(te_acc)

        epoch_loop.set_postfix({
            "train_acc": f"{tr_acc:.4f}",
            "test_acc": f"{te_acc:.4f}",
            "train_loss": f"{tr_loss:.4f}",
            "test_loss": f"{te_loss:.4f}"
        })

    return results

