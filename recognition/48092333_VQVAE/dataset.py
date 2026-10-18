"""HipMRI 2D prostate slices: loading, normalisation and patient-disjoint splits.

Run `python dataset.py` for a data audit (slices, patients and shapes per split).
"""

import argparse
import os
import re
from collections import Counter

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
SPLITS = {"train": "keras_slices_train", "validate": "keras_slices_validate",
          "test": "keras_slices_test"}
IMAGE_SHAPE = (256, 128)


def patient_id(path):
    return re.search(r"case_\d+", os.path.basename(path)).group(0)


def split_paths(data_root=DATA_ROOT):
    """Slice paths per split; fails if any patient appears in more than one split."""
    paths = {}
    for split, folder in SPLITS.items():
        folder = os.path.join(data_root, folder)
        paths[split] = sorted(os.path.join(folder, f) for f in os.listdir(folder) if ".nii" in f)
    patients = [{patient_id(p) for p in ps} for ps in paths.values()]
    if patients[0] & patients[1] or patients[0] & patients[2] or patients[1] & patients[2]:
        raise ValueError("patient leakage between splits")
    return paths


def load_slice(path):
    """Adapted from load_data_2D in Appendix B of the specification."""
    image = nib.load(path).get_fdata(caching="unchanged")
    return (image[:, :, 0] if image.ndim == 3 else image).astype(np.float32)


def fit_to_shape(image, shape=IMAGE_SHAPE):
    """Centre-crop or zero-pad to `shape` (60 slices are 256 x 144)."""
    out = np.zeros(shape, np.float32)
    h, w = min(image.shape[0], shape[0]), min(image.shape[1], shape[1])
    i, j = (image.shape[0] - h) // 2, (image.shape[1] - w) // 2
    k, m = (shape[0] - h) // 2, (shape[1] - w) // 2
    out[k:k + h, m:m + w] = image[i:i + h, j:j + w]
    return out


def normalise(image):
    """Clip to the 0.5-99.5th percentiles and rescale to [0, 1]."""
    lo, hi = np.percentile(image, [0.5, 99.5])
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)


class HipMRISlices(Dataset):
    """Slices held in memory as float16; optional random flip and gamma jitter."""

    def __init__(self, paths, augment=False):
        self.paths = paths
        self.augment = augment
        self.images = torch.empty(len(paths), 1, *IMAGE_SHAPE, dtype=torch.float16)
        for i, path in enumerate(tqdm(paths, leave=False)):
            self.images[i, 0] = torch.from_numpy(normalise(fit_to_shape(load_slice(path))))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = self.images[i].float()
        if self.augment:
            if torch.rand(1) < 0.5:
                x = x.flip(-1)
            x = x ** float(torch.empty(1).uniform_(0.9, 1.1))
        return x


def get_datasets(data_root=DATA_ROOT, max_slices=None, augment=False, splits=tuple(SPLITS), seed=42):
    """Datasets per split; `max_slices` keeps a fixed random subset of each split."""
    generator = torch.Generator().manual_seed(seed)
    datasets = {}
    for split, paths in split_paths(data_root).items():
        if max_slices and len(paths) > max_slices:
            keep = torch.randperm(len(paths), generator=generator)[:max_slices].sort().values
            paths = [paths[i] for i in keep]
        if split in splits:
            datasets[split] = HipMRISlices(paths, augment and split == "train")
    return datasets


def get_dataloaders(batch_size=32, num_workers=4, **kwargs):
    datasets = get_datasets(**kwargs)
    loaders = {s: DataLoader(d, batch_size, shuffle=s == "train", drop_last=s == "train",
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())
               for s, d in datasets.items()}
    return loaders, datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HipMRI data audit")
    parser.add_argument("--data-root", default=DATA_ROOT)
    args = parser.parse_args()
    for split, paths in split_paths(args.data_root).items():
        shapes = Counter(nib.load(p).shape[:2] for p in paths)
        patients = len({patient_id(p) for p in paths})
        print(f"{split}: {len(paths)} slices, {patients} patients, shapes {dict(shapes)}")
    print("No patient appears in more than one split.")