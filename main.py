from models import SimpleCNN
from dataset import SimpleDataset
from trainer import SimpleTrainer
import tools.utils as utils
import torch.utils.data as data
import torch.optim as optim
import torch
from datetime import datetime
import os



# defining parameters
experiment_name = 'tomatoes-segmentation'
ds_path = '/home/giovanni/Desktop/agricultural-robotics/datasets/enhanced-tomato-greenhouse-dataset/'
img_path = os.path.join(ds_path, 'images')
ann_path = os.path.join(ds_path, 'annotations.json')
total_epochs = 10
batch_size = 4
learning_rate = 5e-4

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
checkpoint_path = os.path.join('models', timestamp) + '/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using {device} device')



# setting random seed
utils.set_seed()

# splitting dataset into train, validation and test
train, valid, test = utils.get_dataset_splits(checkpoint_path, ann_path, (0.8, 0.1, 0.1))
print(f'train: {len(train)}, validation: {len(valid)}, test: {len(test)}')

# initializing datasets
state = {'current_epoch': 0, 'total_epochs': total_epochs}
train_ds = SimpleDataset(ann_path, img_path, train, 0, 960, 540, state, augment=True, curriculum=True)
valid_ds = SimpleDataset(ann_path, img_path, valid, 0, 960, 540)
test_ds = SimpleDataset(ann_path, img_path, test, 0, 960, 540)
print('datasets created')

# initializing dataloaders
train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"), num_workers=4)
valid_loader = data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"), num_workers=4)
test_loader = data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"), num_workers=4)
print('dataloaders created')



# initializing model
model = SimpleCNN(in_channels=3).to(device)

# initializing loss function
loss_fn = utils.bce_dice_loss

# initializing optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

hparams={
    "learning_rate": learning_rate,
    "batch_size":    batch_size
}

# initializing trainer
trainer = SimpleTrainer(model, test_loader, valid_loader, optimizer, loss_fn, total_epochs, experiment_name, checkpoint_path=checkpoint_path, hparams=hparams)



# starting training
print('starting training...')
trainer.fit()
print('training completed')

# computing testing loss
test_loss = trainer.evaluate_metrics(train_loader)
print(f'test loss: {test_loss}')