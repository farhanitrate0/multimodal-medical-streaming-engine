"""Disk-backed dataset cache read lazily through memmap."""
import os
import zipfile

import numpy as np
import torch
from PIL import Image

CACHE_DIR = os.environ.get("MEMMAP_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache"))
MEDMNIST_ROOT = os.path.expanduser("~/.medmnist")
CHUNK = 256  # images per write


def _npz_path(flag, size):
    path = os.path.join(MEDMNIST_ROOT, f"{flag}_{size}.npz")
    if not os.path.exists(path):
        import medmnist
        from medmnist import INFO
        getattr(medmnist, INFO[flag]["python_class"])(split="test", download=True, size=size)  # fetch only
    return path


def _copy_member(zf, member, dst):
    """Decompress one .npy from the zip into a memmap, chunk by chunk."""
    with zf.open(member) as src:
        version = np.lib.format.read_magic(src)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read_header(src)
        out = np.lib.format.open_memmap(dst + ".tmp", mode="w+", dtype=dtype, shape=shape, fortran_order=fortran)
        row = int(np.prod(shape[1:])) * dtype.itemsize
        for i in range(0, shape[0], CHUNK):
            n = min(CHUNK, shape[0] - i)
            out[i:i + n] = np.frombuffer(src.read(n * row), dtype=dtype).reshape((n,) + shape[1:])
        out.flush()
        del out
    os.replace(dst + ".tmp", dst)


def build_cache(split, flag="pneumoniamnist", size=224, root=CACHE_DIR):
    """Unpack a split once; later runs reuse it."""
    os.makedirs(root, exist_ok=True)
    imgs = os.path.join(root, f"{flag}_{size}_{split}_images.npy")
    labels = os.path.join(root, f"{flag}_{size}_{split}_labels.npy")
    if not (os.path.exists(imgs) and os.path.exists(labels)):
        with zipfile.ZipFile(_npz_path(flag, size)) as zf:
            _copy_member(zf, f"{split}_images.npy", imgs)
            _copy_member(zf, f"{split}_labels.npy", labels)
    return imgs, labels


class MemmapDataset(torch.utils.data.Dataset):
    """Drop-in for the MedMNIST class; only touched pages are read from disk."""

    def __init__(self, split, transform=None, flag="pneumoniamnist", size=224, root=CACHE_DIR):
        imgs, labels = build_cache(split, flag, size, root)
        self.imgs = np.load(imgs, mmap_mode="r")
        self.labels = np.load(labels)  # tiny, keep in ram
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = Image.fromarray(np.asarray(self.imgs[i]))
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i].astype(np.int64)


def stream_batches(path, batch_size=64):
    """Yield consecutive slices without loading the file."""
    arr = np.load(path, mmap_mode="r")
    for i in range(0, len(arr), batch_size):
        yield np.asarray(arr[i:i + batch_size])
