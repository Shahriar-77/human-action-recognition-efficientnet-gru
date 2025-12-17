
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
from pathlib import Path
from glob import glob
import random

class VideoDataset_1(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_clip=16, frame_skip=2):
        """
        root_dir: path to dataset split (e.g., Data/ETH_Actions/train)
        transform: torchvision transforms to apply to each frame
        frames_per_clip: number of frames to sample per video
        frame_skip: skip factor to spread frames across the video
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.frame_skip = frame_skip

        # Get all videos and labels
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.video_paths = []
        for cls in self.classes:
            for vid in glob(str(self.root_dir / cls / "*.avi")):
                self.video_paths.append((vid, cls))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path, label_name = self.video_paths[idx]
        label = self.class_to_idx[label_name]
        frames = self._load_video(video_path)

        # Convert to PIL for transforms
        if self.transform:
            from torchvision.transforms.functional import to_pil_image
            frames = [self.transform(to_pil_image(f)) for f in frames]

        frames = torch.stack(frames)  # (T, C, H, W)
        frames = frames.mean(dim=0)    # (C, H, W) ← average frames

        return frames, label



    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = list(range(0, total_frames, self.frame_skip))
        if len(frame_indices) > self.frames_per_clip:
            frame_indices = random.sample(frame_indices, self.frames_per_clip)

        frames = []
        for i in sorted(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)
        cap.release()

        # Pad if fewer frames than expected
        while len(frames) < self.frames_per_clip:
            frames.append(frames[-1])
        return frames


class VideoDataset_2(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_clip=16, frame_skip=2):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.frame_skip = frame_skip

        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.video_paths = []
        for cls in self.classes:
            for vid in glob(str(self.root_dir / cls / "*.avi")):
                self.video_paths.append((vid, cls))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path, label_name = self.video_paths[idx]
        label = self.class_to_idx[label_name]
        frames = self._load_video(video_path)

        processed_frames = []
        for f in frames:
            # Convert to grayscale [H, W]
            if f.ndim == 3:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            # Convert to tensor [1, H, W]
            f = torch.from_numpy(f).unsqueeze(0).float() / 255.0
            if self.transform:
                from torchvision.transforms.functional import to_pil_image
                f = self.transform(to_pil_image(f))
            processed_frames.append(f)

        frames_tensor = torch.stack(processed_frames)  # [T, 1, H, W]
        return frames_tensor, label

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = list(range(0, total_frames, self.frame_skip))
        if len(frame_indices) > self.frames_per_clip:
            frame_indices = random.sample(frame_indices, self.frames_per_clip)

        frames = []
        for i in sorted(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(frame)
        cap.release()

        # Pad if fewer frames than expected
        while len(frames) < self.frames_per_clip:
            frames.append(frames[-1])
        return frames



class VideoDataset_3(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_clip=16, frame_skip=1):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.frame_skip = frame_skip

        # Classes and mapping
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Gather video paths
        self.video_paths = []
        for cls in self.classes:
            for vid in glob(str(self.root_dir / cls / "*.avi")):
                self.video_paths.append((vid, cls))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path, label_name = self.video_paths[idx]
        label = self.class_to_idx[label_name]

        frames = self._load_video(video_path)
        processed_frames = []

        for f in frames:
            if f.ndim == 2:  # H x W grayscale
                f = torch.from_numpy(f).unsqueeze(0).float() / 255.0
            else:
                f = torch.from_numpy(f).permute(2,0,1).float() / 255.0  # unlikely for KTH

            if self.transform:
                from torchvision.transforms.functional import to_pil_image
                f = self.transform(to_pil_image(f))
            processed_frames.append(f)

        frames_tensor = torch.stack(processed_frames)  # [T, 1, H, W]
        return frames_tensor, label

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = list(range(0, total_frames, self.frame_skip))

        # Sample frames if longer than frames_per_clip
        if len(frame_indices) > self.frames_per_clip:
            frame_indices = sorted(random.sample(frame_indices, self.frames_per_clip))

        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            # ETH is grayscale
            if len(frame.shape) == 3 and frame.shape[2] > 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)

        cap.release()

        # Pad if fewer frames
        while len(frames) < self.frames_per_clip:
            frames.append(frames[-1])
        return frames



class VideoDataset_4(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_clip=16, frame_skip=1):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.frame_skip = frame_skip

        # Classes and mapping
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Gather video paths
        self.video_paths = []
        for cls in self.classes:
            for vid in glob(str(self.root_dir / cls / "*.avi")):
                self.video_paths.append((vid, cls))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path, label_name = self.video_paths[idx]
        label = self.class_to_idx[label_name]

        frames = self._load_video(video_path)
        processed_frames = []

        for f in frames:
            if f.ndim == 2:
                f = torch.from_numpy(f).unsqueeze(0).float() / 255.0
            else:
                f = torch.from_numpy(f).permute(2,0,1).float() / 255.0

            if self.transform:
                from torchvision.transforms.functional import to_pil_image
                f = self.transform(to_pil_image(f))
            processed_frames.append(f)

        frames_tensor = torch.stack(processed_frames)  # [T, 1, H, W]

        # Return dict for compatibility with training loop
        return {
            "frames": frames_tensor, 
            "labels": torch.tensor(label, dtype=torch.long)
        }


    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = list(range(0, total_frames, self.frame_skip))

        # Sample frames if longer than frames_per_clip
        if len(frame_indices) > self.frames_per_clip:
            frame_indices = sorted(random.sample(frame_indices, self.frames_per_clip))

        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            # KTH is grayscale
            if len(frame.shape) == 3 and frame.shape[2] > 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)

        cap.release()

        # Pad if fewer frames
        while len(frames) < self.frames_per_clip:
            frames.append(frames[-1])
        return frames



