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



# defining parameters
experiment_name = 'tomatoes-segmentation'
ds_path = '/home/giovanni/Desktop/agricultural-robotics/datasets/enhanced-tomato-greenhouse-dataset/'
img_path = os.path.join(ds_path, 'images')
ann_path = os.path.join(ds_path, 'annotations.json')
total_epochs = 500
batch_size = 4
learning_rate = 5e-4
curriculum = True
augment=True
dropout = 0.25
# 0 = tomatoes
target_category = 0



timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
splits_path = os.path.join('models', timestamp)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using {device} device')



# splitting dataset into train, validation and test
train, valid, test = utils.get_dataset_splits(splits_path, ann_path, (0.8, 0.1, 0.1))
print(f'train: {len(train)}, validation: {len(valid)}, test: {len(test)}')

# setting state
state = {'current_epoch': 0, 'total_epochs': total_epochs}
state=state if curriculum else None

# initializing datasets
train_ds = SimpleDataset(ann_path, img_path, train, target_category, 360, 640, state, augment=augment, curriculum=curriculum)
valid_ds = SimpleDataset(ann_path, img_path, valid, target_category, 360, 640)
test_ds = SimpleDataset(ann_path, img_path, test, target_category, 360, 640)
print('datasets created')

# initializing dataloaders
train_loader = data.DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=(device.type == "cuda"),
    num_workers=2,
    persistent_workers=True,
)
valid_loader = data.DataLoader(
    valid_ds,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=(device.type == "cuda"),
    num_workers=2,
    persistent_workers=True,
)
test_loader = data.DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=(device.type == "cuda"),
    num_workers=2,
    persistent_workers=True,
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
        "loss_fn": loss_fn.__name__,
        "epochs": total_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "dropout": dropout_val,
        "augment": augment,
        "curriculum": curriculum,
        "device": device,
    }

    run_name = f"{model_name}_{timestamp}"
    checkpoint_path = os.path.join('models', timestamp, model_name)
    # initializing trainer
    trainer = SimpleTrainer(
        model,
        train_loader,
        valid_loader,
        optimizer,
        loss_fn,
        total_epochs,
        state=state,
        checkpoint_path=checkpoint_path,
        hparams=hparams,
    )



    # starting training

    #setting up mlflow
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
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