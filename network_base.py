import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import *
from training_utils import *

class Network(nn.Module):
    def __init__(self, no_cameras, device):
        super(Network, self).__init__()
        self.device = device
        self.no_cameras = no_cameras
        
        self.conv_layers = nn.ModuleList([ImageLayer(SEQUENCE_LENGTH * NETWORK_IMAGE_LAYER_SIZE).to(device) for _ in range(no_cameras)])
    
        conv_output_size = 256 * no_cameras
        hidden_size = 256

        self.combine_layer = nn.Linear(conv_output_size, hidden_size)

        layer_positions = hidden_size + 7

        self.mlp = nn.Sequential(
            nn.Linear(layer_positions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NETWORK_OUTPUT_SIZE),
        )

    def forward(self, x, positions):
        image_output = torch.tensor([], device=self.device)

        for idx, input in enumerate(x):

            input = self.conv_layers[idx](input)
            
            image_output = torch.concat((image_output, input), dim = 2)

        if len(x) > 1:
            image_output = self.combine_layer(image_output.squeeze(1))
            image_output = torch.relu(image_output)
        
        output = torch.concat((image_output.squeeze(1), positions), dim=1)
        
        return self.mlp(output)


def train() -> None:
    """
    Train the neural network model.
    """
    device = torch.device(DEVICE)
    model = Network(len(ACTIVE_CAMERAS), device).to(device)
    starting_epoch = 0
    epoch_losses: List[float] = []
    validation_losses: List[float] = []

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=0.5)
    print(MODEL_PATH)

    if TRAIN_MODEL_MORE:
        model.load_state_dict(torch.load(f'{MODEL_PATH}_{STARTING_EPOCH}.pth', map_location=DEVICE))
        model.train()
        details = torch.load(DETAILS_PATH)
        starting_epoch = STARTING_EPOCH
        epoch_losses = details["epoch_losses"][:STARTING_EPOCH]
        validation_losses = details["validation_losses"][:STARTING_EPOCH]
        [scheduler.step() for _ in range(starting_epoch)]

    labels = torch.load(SAMPLE_LABEL_PATH)
    labels = prepare_labels(labels)
    dataset = CustomImageDataset(labels)

    train_sampler = EpisodeBatchSampler(dataset.labels, batch_size=BATCH_SIZE, train=True, random=True)
    train_dataloader = DataLoader(dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=True)

    valid_sampler = EpisodeBatchSampler(dataset.labels, batch_size=BATCH_SIZE, train=False, random=False)
    valid_dataloader = DataLoader(dataset, batch_sampler=valid_sampler, num_workers=0, pin_memory=True)

    criterion = nn.MSELoss()
    fig, ax = plt.subplots()
    ax.set(xlabel='Epoch', ylabel='Loss', title='Loss Curve for epochs')
    plt.yscale('log')

    with tqdm(total=EPOCHS_TO_TRAIN * (len(train_dataloader) + len(valid_dataloader))) as pbar:
        for epoch in tqdm(range(EPOCHS_TO_TRAIN), leave=False):
            pbar.write(f'Current Learning Rate: {scheduler.get_last_lr()}')
            display_epoch = starting_epoch + epoch + 1

            train_loss = run_epoch(model, train_dataloader, criterion, pbar, display_epoch, device, is_training=True, optimizer=optimizer)
            epoch_losses.append(train_loss)
            pbar.write(f'LOSS epoch {display_epoch}: {train_loss}')

            train_sampler.shuffle_episodes()
            pbar.write(f'STARTING VALIDATION')
            val_loss = run_epoch(model, valid_dataloader, criterion, pbar, display_epoch, device, is_training=False)
            validation_losses.append(val_loss)
            pbar.write(f'Validation Loss epoch {display_epoch}: {val_loss}')

            torch.save(model.state_dict(), f'{MODEL_PATH}_{display_epoch}.pth')
            details = {"epoch_losses": epoch_losses, "validation_losses": validation_losses, "optimizer": optimizer.state_dict()}
            torch.save(details, DETAILS_PATH)
            scheduler.step()

    plt.plot(range(1, starting_epoch + EPOCHS_TO_TRAIN + 1), epoch_losses, label='Epoch Loss')
    plt.plot(range(1, starting_epoch + EPOCHS_TO_TRAIN + 1), validation_losses, label='Validation Epoch Loss')
    fig.savefig(f'loss_figs/loss_{MODEL_PATH.split("/")[1]}_{MODEL_PATH.split("/")[2]}.png')
    plt.show(block=True)

if __name__ == "__main__":
    if USE_TRANSFORMERS or USE_LSTM:
        raise ValueError("This network type should be used only when both use_transformers and use_lstm are false.")
    
    os.makedirs(f'models/{TASK_TYPE.name.lower()}', exist_ok=True)
    os.makedirs(f'loss_figs', exist_ok=True)
    train()