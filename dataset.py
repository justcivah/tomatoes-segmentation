from torch.utils import data
from contextlib import redirect_stdout
from PIL import Image
import tools.utils as utils
from pycocotools.coco import COCO
import torch
import torchvision.transforms.v2.functional as TF
import os
import numpy as np

class SimpleDataset(data.Dataset):
    """
    PyTorch Dataset for binary semantic segmentation with COCO-format annotations.

    Loads images and generates binary foreground/background masks for a single target category. Annotations are decoded via the COCO API and merged into a single binary mask with values in {0, 1}. Images with no annotations receive an all-zero mask rather than being excluded.

    Optionally supports:
      - Random data augmentation, applied per-sample at training time.
      - Curriculum learning, where augmentation intensity is modulated by the current training progress supplied through a shared ``state`` dict.

    Args:
        ann_path (str): Path to the COCO-format annotation JSON file.
        img_path (str): Root directory containing the image files. File names
            are resolved relative to this path using the ``file_name`` field
            stored in the COCO metadata.
        ids (list[int]): Ordered list of COCO image IDs to expose as dataset
            samples. The dataset length and index mapping are derived from this
            list.
        cat (int): COCO category ID of the target class. Only annotations
            belonging to this category are used when building masks.
        width (int): Width to which every image and mask is resized
            before being returned.
        height (int): Height to which every image and mask is
            resized before being returned.
        state (dict | None): Mutable dictionary shared with the training loop.
            Required when ``curriculum=True``, ignored otherwise. Must contain:
            - 'current_epoch' (int): Zero-based index of the current epoch.
            - 'total_epochs' (int): Total number of planned training epochs.

            Because the dict is shared by reference, updates made by the
            training loop are automatically visible inside the dataset.
        augment (bool): If ``True``, applies random spatial and photometric
            augmentations to each sample via :func:`augmentations`.
            Default: ``False``.
        curriculum (bool): If ``True``, passes the current training progress
            from ``state`` to :func:`augmentations`, allowing augmentation
            intensity to be scheduled over the course of training.
            Requires ``state`` to be provided; raises :exc:`ValueError`
            otherwise. Default: ``False``.

    Raises:
        ValueError: If ``curriculum=True`` and ``state`` is ``None``.
    """
    
    def __init__(self, ann_path: str, img_path: str, ids: list[int], cat: int, height: int, width: int, state: dict | None = None, augment: bool = False, curriculum: bool = False):

        self.ann_path = ann_path
        self.img_path = img_path
        self.ids = ids
        self.cat = cat
        self.height = height
        self.width = width
        self.state = state
        self.augment = augment
        self.curriculum = curriculum

        if self.curriculum:
            if state is None:
                raise ValueError("state must be provided when curriculum=True")

        with redirect_stdout(open(os.devnull, 'w')):
            self.coco = COCO(self.ann_path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(img_id, self.cat)
        ann = self.coco.loadAnns(ann_ids)

        img = Image.open(os.path.join(self.img_path, self.coco.imgs[img_id]['file_name']))

        # if there are no annotations for an image, create an all-zeros matrix
        if len(ann) == 0:
            h = self.coco.imgs[img_id]['height']
            w = self.coco.imgs[img_id]['width']
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = sum(self.coco.annToMask(x) for x in ann)
            mask = np.clip(mask, 0, 1)

        # convert image and mask to float32 torch tensors
        img = TF.to_image(img)
        img = TF.to_dtype(img, torch.uint8, scale=True)

        mask = TF.to_image(mask)
        mask = TF.to_dtype(mask, torch.uint8, scale=False)

        img, mask = utils.transformations(img, mask, self.height, self.width)
        
        if self.augment:
            img, mask = utils.augmentations(img, mask, self.height, self.width, self.state, self.curriculum)

        img = TF.to_dtype(img, torch.float32, scale=True)
        mask = TF.to_dtype(mask, torch.float32, scale=False)

        return img, mask