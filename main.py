import models as md
from dataset import SimpleDataset
from trainer import SimpleTrainer
import tools.utils as utils
import torch.utils.data as data
import torch.optim as optim
import torch
import mlflow
from datetime import datetime
from functools import partial
import os
from config import (
    DS_PATH as ds_path,
    EXPERIMENT_NAME as experiment_name,
    TARGET_CATEGORY as target_category,
    IMG_HEIGHT as img_height,
    IMG_WIDTH as img_width,
    NUM_WORKERS as num_workers,
    TOTAL_EPOCHS as total_epochs,
    BATCH_SIZE as batch_size,
    LEARNING_RATE as learning_rate,
    CURRICULUM as curriculum,
    AUGMENT as augment,
    DROPOUT as dropout,
    PATIENCE as patience,
)


img_path = os.path.join(ds_path, 'images')
ann_path = os.path.join(ds_path, 'annotations.json')

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
splits_path = os.path.join('models', timestamp)
experiment_name = experiment_name + '_' + timestamp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using {device} device')



# splitting dataset into train, validation and test
train, valid, test = utils.get_dataset_splits(splits_path, ann_path, (0.8, 0.1, 0.1))
print(f'train: {len(train)}, validation: {len(valid)}, test: {len(test)}')

# setting state
state = {'current_epoch': 0, 'total_epochs': total_epochs}
state=state if curriculum else None

# initializing datasets
train_ds = SimpleDataset(ann_path, img_path, train, target_category, img_height, img_width, state, augment=augment, curriculum=curriculum)
valid_ds = SimpleDataset(ann_path, img_path, valid, target_category, img_height, img_width)
test_ds = SimpleDataset(ann_path, img_path, test, target_category, img_height, img_width)
print('datasets created')

# initializing dataloaders
train_loader = data.DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=(device.type == "cuda"),
    num_workers=num_workers,
)
valid_loader = data.DataLoader(
    valid_ds,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=(device.type == "cuda"),
    num_workers=num_workers,
)
test_loader = data.DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=(device.type == "cuda"),
    num_workers=num_workers,
)
print('dataloaders created')



# multiple model train
models = [
    md.SimpleCNN,
    md.CNN,
    partial(md.DropoutCNN, dropout=dropout),
    partial(md.Dropout2DCNN, dropout=dropout),
    partial(md.EDPoolingCNN, dropout=dropout),
    partial(md.DoubleEDPoolingCNN, dropout=dropout),
    partial(md.EDStridingCNN, dropout=dropout),
    partial(md.EDSplitStridingCNN, dropout=dropout),
    partial(md.DoubleEDCNN, dropout=dropout),
    partial(md.SkipDoubleEDCNN, dropout=dropout),
    partial(md.SkipBothDoubleEDCNN, dropout=dropout),
    partial(md.LightSkipBothDoubleEDCNN, dropout=dropout),
    partial(md.LightFuseSkipBothDoubleEDCNN, dropout=dropout),
]

# initializing loss function
loss_fn = utils.bce_dice_loss



#setting up mlflow
mlflow.set_experiment(experiment_name)
# train all models
for m in models:

    # setting random seed for each model run
    utils.set_seed(42)

    # initializing model
    model = m(in_channels=3).to(device)

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

    checkpoint_path = os.path.join('models', timestamp, model_name)
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
        state=state,
        checkpoint_path=checkpoint_path,
        hparams=hparams,
    )



    # starting training
    with mlflow.start_run(run_name=model_name):
        print(f'starting training for {model_name}...')
        trainer.fit()
        print('training completed')

        # reloading best checkpoint before testing
        best_ckpt = torch.load(os.path.join(checkpoint_path, 'best_model.pt'), weights_only=True)
        model.load_state_dict(best_ckpt['model_state_dict'])

        # computing testing loss
        test_metrics = trainer.evaluate_metrics(test_loader)
        print(f'test metrics: {test_metrics}')
        # logging test metrics to mlflow
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # clearing objects and cache for next run
        del model
        del optimizer
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()