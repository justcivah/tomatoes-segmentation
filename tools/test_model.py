import os
import model as md
from dataset import SimpleDataset
from trainer import SimpleTrainer
import tools.utils as utils
import torch.utils.data as data
import torch.optim as optim
import torch
from functools import partial
import gc
from config import (
    DS_PATH as ds_path,
    EXPERIMENT_NAME as experiment_name,
    TARGET_CATEGORY as target_category,
    IMG_HEIGHT as img_height,
    IMG_WIDTH as img_width,
    NUM_WORKERS as num_workers,
    SEED as seed,
    TOTAL_EPOCHS as total_epochs,
    BATCH_SIZE as batch_size,
    LEARNING_RATE as learning_rate,
    CURRICULUM as curriculum,
    AUGMENT as augment,
    DROPOUT as dropout,
    PATIENCE as patience,
    EARLY_STOP_START_EPOCH as early_stop_start_epoch,
)



# setting seed
utils.set_seed(seed)

# print config data
print(f'ds_path: {ds_path}')
print(f'img_height: {img_height}')
print(f'img_width: {img_width}')



img_path = os.path.join(ds_path, 'images')
msk_path = os.path.join(ds_path, 'masks')
ann_path = os.path.join(ds_path, 'annotations.json')

timestamp = '2026-04-11_02-42-30'
splits_path = os.path.join('../models', timestamp)
experiment_name = experiment_name + '_' + timestamp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using {device} device')



# splitting dataset into train, validation and test
train, valid, test = utils.get_dataset_splits(splits_path, ann_path, (0.05, 0.05, 0.9))
print(f'train: {len(train)}, validation: {len(valid)}, test: {len(test)}')

# setting state
state = {'current_epoch': 0, 'total_epochs': total_epochs}
state=state if curriculum else None

# initializing datasets
train_ds = SimpleDataset(ann_path, img_path, msk_path, train, target_category, img_height, img_width, state, augment=augment, curriculum=curriculum)
valid_ds = SimpleDataset(ann_path, img_path, msk_path, valid, target_category, img_height, img_width)
test_ds = SimpleDataset(ann_path, img_path, msk_path, test, target_category, img_height, img_width)
print('datasets created')


train_generator = torch.Generator().manual_seed(seed)
valid_generator = torch.Generator().manual_seed(seed)
test_generator  = torch.Generator().manual_seed(seed)
# initializing dataloaders
train_loader = data.DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    generator=train_generator,
    pin_memory=(device.type == "cuda"),
    worker_init_fn=utils.seed_worker,
    num_workers=num_workers,
    prefetch_factor=1 if num_workers > 0 else None,
)
valid_loader = data.DataLoader(
    valid_ds,
    batch_size=batch_size,
    shuffle=False,
    generator=valid_generator,
    pin_memory=(device.type == "cuda"),
    worker_init_fn=utils.seed_worker,
    num_workers=num_workers,
    persistent_workers=(num_workers > 0),
    prefetch_factor=1 if num_workers > 0 else None,
)
test_loader = data.DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    generator=test_generator,
    pin_memory=(device.type == "cuda"),
    worker_init_fn=utils.seed_worker,
    num_workers=num_workers,
    prefetch_factor=1 if num_workers > 0 else None,
)
print('dataloaders created')



# multiple model train
models = [
    partial(md.YNet, dropout=dropout),
]

# initializing loss function
loss_fn = utils.bce_dice_loss



# train all models
for m in models:
    # setting random seed for each model run
    utils.set_seed(42)

    # initializing model
    model = m(in_channels=3).to(device)
    model = torch.compile(model)

    # initializing optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # setting hyperparameters log
    model_class = m.func if isinstance(m, partial) else m
    model_name  = model_class.__name__
    dropout_val = m.keywords.get('dropout', None) if isinstance(m, partial) else None

    hparams = {
        "model": model_name,
        "optimizer": optimizer.__class__.__name__,
        "img_height": img_height,
        "img_width": img_width,
        "loss_fn": loss_fn.__name__,
        "epochs": total_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "dropout": dropout_val,
        "patience": patience,
        "augment": augment,
        "curriculum": curriculum,
        "device": str(device),
    }

    checkpoint_path = os.path.join('models', '2026-04-11_02-42-30', model_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    # initializing trainer
    trainer = SimpleTrainer(
        model,
        train_loader,
        valid_loader,
        optimizer,
        loss_fn,
        total_epochs,
        patience=patience,
        early_stop_start_epoch=early_stop_start_epoch,
        state=state,
        checkpoint_path=checkpoint_path,
        hparams=hparams,
    )



    # reloading best checkpoint before testing
    best_ckpt = torch.load(os.path.join(checkpoint_path, 'best_model.pt'), weights_only=True, map_location='cpu')
    model.load_state_dict(best_ckpt['model_state_dict'])

    # computing testing loss
    test_metrics = trainer.evaluate_metrics(test_loader)
    print(f'test metrics: {test_metrics}')

    # clearing objects and cache for next run
    del model
    del optimizer
    del trainer
    del best_ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()