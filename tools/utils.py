import torchvision.transforms.v2 as T
from torchvision import tv_tensors
import torch.nn as nn
import torch
import math

from contextlib import redirect_stdout
from pycocotools.coco import COCO
import numpy as np
import random
import math
import json
import os

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset_splits(splits_output: str, annotations_path: str, splits: tuple = (0, 0, 0)):
    """
    Returns three lists containing integers representing the image IDs from the COCO annotations file. These IDs are randomly shuffled and then split according to the probabilities specified in the splits tuple (train, validation, test).

    All the elements in the tuple must sum up to 1. If one or more elements are set to zero, the related lists will be empty.

    Args:
        splits_output (str): Path where the split IDs will be saved.
        annotations_path (str): Path to the COCO annotations file.
        splits (tuple): Tuple of three floats representing the proportions for
                        (train, validation, test). Must sum to 1.

    Returns:
        tuple: Three lists containing image IDs for train, validation, and test splits.

    Raises:
        ValueError: If ``splits`` does not contain 3 elements, or the elements do not sum to 1 with a tolerance of 1e-6.
    """

    if len(splits) != 3:
        raise ValueError(f"splits must be a tuple of 3 elements, got {len(splits)}")
    if abs(sum(splits) - 1) > 1e-6:
        raise ValueError(f"splits must sum to 1, got {sum(splits):.6f}")
    
    os.makedirs(splits_output, exist_ok=True)

    with redirect_stdout(open(os.devnull, 'w')):
        coco = COCO(annotations_path)
    ids = coco.getImgIds()

    random.shuffle(ids)

    n_tot = len(ids)
    n_train = math.ceil(splits[0] * n_tot)
    n_valid = math.floor(splits[1] * n_tot)

    train = ids[:n_train]
    valid = ids[n_train:n_train + n_valid]
    test = ids[n_train + n_valid:]

    splits_dict = {'train': train, 'valid': valid, 'test': test}
    with open(os.path.join(splits_output, 'splits.json'), 'w') as f:
        json.dump(splits_dict, f, indent=4)

    return train, valid, test


def transformations(img, mask, t_height, t_width):
    h, w = img.shape[-2:]

    scale = max(t_width / w, t_height / h)
    
    # Use math.ceil to prevent truncation errors
    n_height = math.ceil(h * scale)
    n_width = math.ceil(w * scale)

    transform = T.Compose([
        T.Resize((n_height, n_width), antialias=True),
        T.RandomCrop((t_height, t_width)),
    ])

    img, mask = transform(img, tv_tensors.Mask(mask))
    return img, mask


def compute_augmentation_values(state, curriculum):
    config_dir = os.path.dirname(os.path.abspath(__file__))

    if curriculum:
        with open(os.path.join(config_dir, 'augmentation_curriculum_config_revision.json'), 'r') as f:
            config = json.load(f)

        progress = min(state['current_epoch'] / (state['total_epochs'] * 0.8), 1)
        return {
            transform: {k: v['s'] + (v['e'] - v['s']) * progress for k, v in params.items()}
            for transform, params in config.items()
        }
    
    else:
        with open(os.path.join(config_dir, 'augmentation_config.json'), 'r') as f:
            return json.load(f)
        

def augmentations(img, mask, height, width, state, curriculum):
    v = compute_augmentation_values(state, curriculum)

    # the blur kernel must be an odd integer
    k = int(v['blur']['k'])
    k = k if k % 2 == 1 else k + 1

    spatial = T.Compose([
        T.RandomApply([T.RandomCrop(size=(int(height * v['crop']['s1']), int(width * v['crop']['s2']))), T.Resize((height, width))], p=v['crop']['p']),
        T.RandomHorizontalFlip(p=v['h_flip']['p']),
        T.RandomVerticalFlip(p=v['v_flip']['p']),
        T.RandomApply([T.RandomRotation(degrees=(-max(v['rotation']['d'], 1), max(v['rotation']['d'], 1)))], p=v['rotation']['p']),
        T.RandomApply([T.RandomZoomOut(side_range=(v['zoomout']['s1'], v['zoomout']['s2']), p=1), T.Resize((height, width))], p=v['zoomout']['p']),
        T.RandomApply([T.ElasticTransform(alpha=v['elastic']['a'])], p=v['elastic']['p']),
        T.RandomApply([T.RandomAffine(degrees=(0, 0), translate=(v['translate']['t1'], v['translate']['t2']))], p=v['translate']['p']),
        T.RandomPerspective(distortion_scale=v['perspective']['s'], p=v['perspective']['p'])
    ])

    photometric = T.Compose([
        T.RandomGrayscale(p=v['grayscale']['p']),
        T.RandomApply([T.ColorJitter(brightness=v['jitter']['b'], contrast=v['jitter']['c'], saturation=v['jitter']['s'], hue=v['jitter']['h'])], p=v['jitter']['p']),
        T.RandomAdjustSharpness(sharpness_factor=v['sharpness']['f'], p=v['sharpness']['p']),
        T.RandomApply([T.GaussianBlur(kernel_size=k)], p=v['blur']['p']),
    ])

    img, mask = spatial(img, tv_tensors.Mask(mask))
    img = photometric(img)

    return img, mask

def bce_dice_loss(pred, target, alpha=0.5):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    
    pred_sigmoid = torch.sigmoid(pred)
    intersection = (pred_sigmoid * target).sum(dim=(1, 2, 3))
    union = pred_sigmoid.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
    
    return alpha * bce + (1 - alpha) * dice.mean()

class EarlyStop():
    def __init__(self, patience=20, start_from_epoch=0, delta=0.001):
        self.patience = patience
        self.start_from_epoch = start_from_epoch
        self.delta = delta
        self.best_loss = float('inf')
        self.count = 0

    def stop(self, loss, epoch):
        if epoch >= self.start_from_epoch:
            if loss < self.best_loss - self.delta:
                self.best_loss = loss
                self.count = 0
            
            else:
                self.count += 1

            if self.count > self.patience:
                return True
            
            return False
        
        else:
            return False