"""
dataset.py - data loading, patient-level splitting and augmentation for HipMRI segmentation.

The Nifti loaders (to_channels, load_data_2D, load_data_3D) are the course-provided code from
Appendix B. Everything under "Added by me" at the bottom is the layer built on top of them.
One line in load_data_2D is also mine (marked ADDED): it resizes the few slices whose size
differs from the first file, which otherwise crash the pre-allocated array.
"""

import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c : c + 1][arr == c] = 1
    return res


# load medical image functions
def load_data_2D(
    imageNames,
    normImage=False,
    categorical=False,
    dtype=np.float32,
    getAffines=False,
    early_stop=False,
):
    """
    Load medical image data from names, cases list provided into a list for
    each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive
    memory usage.

    normImage: bool (normalise the image 0.0-1.0)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for
    quick loading and testing scripts.
    """

    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching="unchanged")
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching="unchanged")  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data
        if inImage.shape != (rows, cols):  # ADDED: a few slices are 256x144, not 256x128
            inImage = resize_2d(inImage, (rows, cols), is_label=(dtype == np.uint8))
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


def load_data_3D(
    imageNames,
    normImage=False,
    categorical=False,
    dtype=np.float32,
    getAffines=False,
    orient=False,
    early_stop=False,
):
    """
    Load medical image data from names, cases list provided into a list for
    each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive
    memory usage.

    normImage: bool (normalise the image 0.0-1.0)
    orient: Apply orientation and resample image? Good for images with large
    slice thickness or anisotropic resolution
    dtype: Type of the data. If dtype=np. uint8, it is assumed that the data
    is labels
    early_stop: Stop loading pre-maturely? Leaves arrays mostly empty, for
    quick loading and testing scripts.
    """

    affines = []

    interp = "linear"
    if dtype == np.uint8:  # assume labels
        interp = "nearest"

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    first_case = niftiImage.get_fdata(caching="unchanged")
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching="unchanged")  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth]  # clip slices
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[
                i, : inImage.shape[0], : inImage.shape[1], : inImage.shape[2], :
            ] = inImage
        else:
            images[
                i, : inImage.shape[0], : inImage.shape[1], : inImage.shape[2]
            ] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images



NUM_CLASSES = 6  # labels 0..5 including background (checked on the real data)


def resize_2d(arr, size, is_label=False):
    """Resize a 2D array to size=(rows, cols). Labels use nearest-neighbour so class
    values are never blended; images use bilinear."""
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None, None]
    if is_label:
        t = torch.nn.functional.interpolate(t, size=size, mode="nearest")
    else:
        t = torch.nn.functional.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def patient_id(path):
    """'case_004_week_0_slice_3.nii.gz' / 'seg_004_...' -> '004'; 'B006_Week0_LFOV.nii.gz' -> 'B006'."""
    parts = os.path.basename(path).split("_")
    return parts[1] if parts[0].lower() in ("case", "seg") else parts[0]


def check_no_leakage(train_files, val_files, test_files):
    """Raise if any patient has files in more than one split."""
    sets = [{patient_id(f) for f in files} for files in (train_files, val_files, test_files)]
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        if sets[i] & sets[j]:
            raise AssertionError(f"Patient leakage between splits: {sorted(sets[i] & sets[j])}")
    print("No patient overlap. Patients train/val/test:", *(len(s) for s in sets))


def get_files_2d(root, split):
    """Sorted (image_files, label_files) for split in 'train' / 'validate' / 'test'.

    root is the keras_slices_data folder. The course already provides the three splits.
    """
    img_dir = os.path.join(root, f"keras_slices_{split}")
    seg_dir = os.path.join(root, f"keras_slices_seg_{split}")
    images = sorted(os.listdir(img_dir))
    images = [os.path.join(img_dir, f) for f in images]
    labels = [os.path.join(seg_dir, os.path.basename(f).replace("case_", "seg_", 1)) for f in images]
    return images, labels


def get_files_3d(root, ratios=(0.7, 0.15, 0.15)):
    """3D volumes come in a single folder, so split by patient here (all weeks stay together).

    Returns {'train': (images, labels), 'validate': ..., 'test': ...}.
    """
    img_dir = os.path.join(root, "semantic_MRs")
    seg_dir = os.path.join(root, "semantic_labels_only")
    images = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
    labels = [os.path.join(seg_dir, os.path.basename(f).replace("LFOV", "SEMANTIC")) for f in images]
    patients = sorted({patient_id(f) for f in images})
    np.random.RandomState(42).shuffle(patients)  # fixed seed -> reproducible split
    n_train, n_val = round(ratios[0] * len(patients)), round(ratios[1] * len(patients))
    owner = {}
    for k, p in enumerate(patients):
        owner[p] = "train" if k < n_train else "validate" if k < n_train + n_val else "test"
    out = {s: ([], []) for s in ("train", "validate", "test")}
    for img, lab in zip(images, labels):
        out[owner[patient_id(img)]][0].append(img)
        out[owner[patient_id(img)]][1].append(lab)
    check_no_leakage(*(out[s][0] for s in ("train", "validate", "test")))
    return out


class SegDataset(Dataset):
    """Wraps arrays from load_data_2D / load_data_3D.

    images: (N, *spatial) float, labels: (N, *spatial) integer class indices (NOT one-hot;
    load them with categorical=False and use F.one_hot where a one-hot target is needed).
    Returns image (1, *spatial) float32 and mask (*spatial) int64.

    flip_axis: which spatial axis to mirror in augmentation. It MUST be the patient's
    left-right axis; flipping the anterior-posterior or superior-inferior axis would swap
    anatomy (e.g. bladder behind the rectum). For the 2D HipMRI slices (256 x 128) the long
    axis, 0, is left-right (checked by plotting a slice). For 3D, check with a plot first.
    """

    def __init__(self, images, labels, augment=False, flip_axis=0):
        self.x = torch.from_numpy(images).unsqueeze(1)
        self.y = torch.from_numpy(labels)
        self.augment = augment
        self.flip_axis = flip_axis

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x, y = self.x[i], self.y[i].long()
        if self.augment:
            if torch.rand(1) < 0.5:  # mirror image and mask together along left-right
                x, y = x.flip(self.flip_axis + 1), y.flip(self.flip_axis)  # +1: x has a channel dim
            if torch.rand(1) < 0.5:  # intensity jitter on the image only
                x = x * (0.9 + 0.2 * torch.rand(1)) + 0.2 * (torch.rand(1) - 0.5)
        return x, y


def get_dataloaders_2d(root, batch_size=16, num_workers=0, early_stop=False):
    """Train/val/test loaders for the 2D slices, using the course's provided splits."""
    files = {s: get_files_2d(root, s) for s in ("train", "validate", "test")}
    check_no_leakage(*(files[s][0] for s in ("train", "validate", "test")))
    loaders = []
    for split in ("train", "validate", "test"):
        imgs, labs = files[split]
        x = load_data_2D(imgs, normImage=True, early_stop=early_stop)
        y = load_data_2D(labs, dtype=np.uint8, early_stop=early_stop)
        loaders.append(DataLoader(SegDataset(x, y, augment=(split == "train")),
                                  batch_size=batch_size, shuffle=(split == "train"),
                                  num_workers=num_workers))
    return loaders


if __name__ == "__main__":
    import sys

    train_loader, val_loader, test_loader = get_dataloaders_2d(sys.argv[1], early_stop=True)
    x, y = next(iter(train_loader))
    print("image", tuple(x.shape), x.dtype, "| mask", tuple(y.shape), y.dtype, "| labels", y.unique().tolist())