"""Datasets, augmentations and annotation I/O.

All datasets are converted (see tools/) to one simple JSON format:

{
  "classes": ["aeroplane", ...],
  "images": [
    {"file": "/abs/path.jpg", "width": 500, "height": 375,
     "boxes": [[x1, y1, x2, y2], ...],   # pixels, original image size
     "labels": [0, 14, ...],             # 0-based class index
     "difficult": [0, 1, ...],           # optional (VOC "difficult" flag)
     "scores": [0.93, ...],              # only in pseudo-label files
     "tags": [0, 14]}                    # image-level tags (classes present)
  ]
}
"""
import json
import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]
CITYSCAPES_CLASSES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


# ----------------------------------------------------------------------------- augmentation
_colour = T.Compose([
    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.RandomApply([T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
])
_erase = [T.RandomErasing(p=p, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random")
          for p in (0.7, 0.5, 0.3)]


def strong_augment(img):
    """Photometric strong augmentation (Unbiased/Adaptive Teacher style).

    It never moves pixels, so boxes stay valid for the augmented image.
    img: float tensor [3, H, W] in [0, 1].
    """
    img = _colour(img)
    for erase in _erase:
        img = erase(img)
    return img.clamp(0, 1)


# ----------------------------------------------------------------------------- dataset
def resize_scale(w, h, min_size, max_size):
    return min(min_size / min(w, h), max_size / max(w, h))


class DetectionDataset(Dataset):
    """Reads the common JSON format.

    train=True  -> random horizontal flip, degenerate boxes removed.
    strong=True -> returns (weak_img, strong_img, target) instead of (img, target).
    use_boxes   -> False for target images whose boxes must not be used (tags only).
    score_thresh-> keep only boxes with score >= thresh (pseudo-label files).

    Images are resized here (shorter side min_size, longer side <= max_size) so data
    loading and augmentation stay cheap. Boxes are scaled with the image; target["scale"]
    maps back to original pixels.
    """

    def __init__(self, ann, train=False, strong=False, min_size=600, max_size=1000,
                 use_boxes=True, score_thresh=None):
        data = load_json(ann) if isinstance(ann, str) else ann
        self.classes = data["classes"]
        self.images = data["images"]
        self.train, self.strong = train, strong
        self.min_size, self.max_size = min_size, max_size
        self.use_boxes, self.score_thresh = use_boxes, score_thresh

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        rec = self.images[i]
        img = Image.open(rec["file"]).convert("RGB")
        w, h = img.size
        s = resize_scale(w, h, self.min_size, self.max_size)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        img = img.resize((nw, nh), Image.BILINEAR)

        if self.use_boxes:
            boxes = torch.tensor(rec.get("boxes", []), dtype=torch.float32).reshape(-1, 4) * s
            labels = torch.tensor(rec.get("labels", []), dtype=torch.int64) + 1  # 0 = background
            difficult = torch.tensor(rec.get("difficult", [0] * len(labels)), dtype=torch.bool)
            if self.score_thresh is not None and "scores" in rec:
                keep = torch.tensor(rec["scores"], dtype=torch.float32) >= self.score_thresh
                boxes, labels, difficult = boxes[keep], labels[keep], difficult[keep]
        else:
            boxes = torch.zeros((0, 4))
            labels = torch.zeros((0,), dtype=torch.int64)
            difficult = torch.zeros((0,), dtype=torch.bool)

        flipped = False
        if self.train and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = boxes.clone()
            boxes[:, [0, 2]] = nw - boxes[:, [2, 0]]
            flipped = True

        if self.train and len(boxes):
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, nw)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, nh)
            ok = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)
            boxes, labels, difficult = boxes[ok], labels[ok], difficult[ok]

        tags = torch.zeros(len(self.classes), dtype=torch.bool)
        if rec.get("tags") is not None:
            tags[torch.tensor(rec["tags"], dtype=torch.int64)] = True

        target = {"boxes": boxes, "labels": labels, "difficult": difficult, "tags": tags,
                  "scale": s, "index": i, "flipped": flipped}
        img = TF.to_tensor(img)
        if self.strong:
            return img, strong_augment(img.clone()), target
        return img, target


def collate(batch):
    return tuple(zip(*batch))


def make_loader(ds, batch_size, workers, shuffle):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      collate_fn=collate, drop_last=shuffle, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0)


def infinite(loader):
    while True:
        for batch in loader:
            yield batch
