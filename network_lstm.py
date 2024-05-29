import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import get_image_path
from torchsummary import summary
import math

from config import *

class ImageLayer(nn.Module):
    def __init__(self):
        super(ImageLayer, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=(3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=2, padding=1)
        self.conv6 = nn.Conv2d(192, 256, kernel_size=(3, 3), stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=2, padding=1)

    def forward(self, x):
        

        # Apply convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.relu(self.conv7(x))

        # Reshape back to (batch_size, number_of_images, -1)

        batch_size, features, height, width = x.size()

        x=x.view(batch_size, height * width, features)

        return x

class Network(nn.Module):
    def __init__(self, no_cameras, device):
        super(Network, self).__init__()
        self.device = device
        self.no_cameras = no_cameras
        
        self.conv_layers = nn.ModuleList([ImageLayer().to(device) for _ in range(no_cameras)])

        self.conv_output_size = 256 * no_cameras
        self.ff_hidden_size = 128 * no_cameras
        last_layer_size = self.ff_hidden_size + 7

        self.lstm = nn.LSTM(input_size=self.conv_output_size, hidden_size=self.ff_hidden_size, num_layers=LSTM_LAYERS, batch_first=True)
        self.fc = nn.Linear(last_layer_size, 15)  # 7 velocities + 2 gripper action + 3 cube positions + 3 end effector positions

        # self.h_0 = torch.randn(LSTM_LAYERS, 1, self.ff_hidden_size).to(self.device)
        # self.c_0 = torch.randn(LSTM_LAYERS, 1, self.ff_hidden_size).to(self.device)

        # self.h_0 = nn.Parameter(self.h_0, requires_grad=True)
        # self.c_0 = nn.Parameter(self.c_0, requires_grad=True)
        self.h_last = 0
        self.reset_hidden_states()

    def reset_hidden_states(self):
        self.h_last = torch.zeros(LSTM_LAYERS, 1, self.ff_hidden_size, requires_grad=False).to(self.device) 
        self.c_last = torch.zeros(LSTM_LAYERS, 1, self.ff_hidden_size, requires_grad=False).to(self.device)

    def forward(self, x, positions):
        image_output = torch.tensor([], device=self.device)
        for idx, input in enumerate(x):
            input = self.conv_layers[idx](input)
            image_output = torch.concat((image_output, input), dim = 2)
        
        lstm_out, (h_last, c_last) = self.lstm(image_output, (self.h_last, self.c_last))

        self.h_last = h_last.clone()
        self.c_last = c_last.clone()
        output = lstm_out[:, -1, :]


        output = torch.concat((output, positions), dim=1)
        return self.fc(output)
    
class EpisodeBatchSampler(Sampler):
    def __init__(self, label_file, batch_size, model: Network, split=0.85, train=True, random=False):
        self.labels = torch.load(label_file)
        self.batch_size = batch_size
        self.train = train
        self.random = random
        self.model = model

        # Splitting the episodes into training and validation sets
        episodes = list(self.labels.keys())
        split_idx = int(len(episodes) * split)
        
        if train:
            self.episodes = episodes[:split_idx]
        else:
            self.episodes = episodes[split_idx:]

        # Map (episode, sample) pairs to dataset indices
        self.dataset_indices = self._create_dataset_indices()
        self.shuffle_episodes()

    def shuffle_episodes(self):
        np.random.shuffle(self.episodes)

    def _create_dataset_indices(self):
        episode_sample_pairs = [
            (episode, sample) for episode, samples in self.labels.items() for sample in samples.keys()
        ]
        return {pair: idx for idx, pair in enumerate(episode_sample_pairs)}

    def __iter__(self):
        batch = []
        for episode in self.episodes:
            samples = list(self.labels[episode].keys())
            if self.random:
                np.random.shuffle(samples)
            for sample in samples:
                idx = self.dataset_indices[(episode, sample)]
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0:
                yield batch
                batch = []
            self.model.reset_hidden_states()

    def __len__(self):
        total_samples = sum(math.ceil(len(list(self.labels[episode].keys())) / self.batch_size) \
                             for episode in self.episodes)
        return total_samples


class CustomImageDataset(Dataset):
    def __init__(self, label_file, device):
        self.labels = torch.load(label_file)
        self.device = device
        self.cameras = ACTIVE_CAMERAS

        self.episode_sample_pairs = [
            (episode, sample) for episode, samples in self.labels.items() for sample in samples.keys()
        ]

    def __len__(self):
        return len(self.episode_sample_pairs)

    def __getitem__(self, idx):
        episode, sample = self.episode_sample_pairs[idx]
        inputs = []
        for camera in self.cameras:
            image_path = get_image_path(camera, episode, sample)
            img = read_image(image_path).to(self.device, dtype=torch.float)
            inputs.append(img)
        
        label = torch.tensor(self.labels[episode][sample]["labels"]).to(self.device)
        positions = torch.tensor(self.labels[episode][sample]["positions"]).to(self.device)
        return inputs, positions, label


def calculate_loss(criterion, outputs, labels):
    loss_1 = criterion(outputs[:, :7], labels[:, :7])
    loss_2 = criterion(outputs[:, 7:9], labels[:, 7:9])
    loss_3 = criterion(outputs[:, 9:12], labels[:, 9:12])
    loss_4 = criterion(outputs[:, 12:15], labels[:, 12:15])
    return loss_1 + loss_2 + loss_3 + loss_4


def train():
    device = torch.device(DEVICE)

    model = Network(len(ACTIVE_CAMERAS), device).to(device)

    starting_epoch = 0

    epoch_losses = []
    batch_losses = []
    validation_losses = []

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=0.5)

    if TRAIN_MODEL_MORE:
        model.load_state_dict(torch.load(MODEL_PATH + '_' + str(STARTING_EPOCH) + '.pth', map_location=DEVICE))
        model.train()

        details = torch.load(DETAILS_PATH)
        
        starting_epoch = STARTING_EPOCH
        epoch_losses = details["epoch_losses"]
        validation_losses = details["validation_losses"]

        [scheduler.step() for _ in range(starting_epoch)]

    dataset = CustomImageDataset(SAMPLE_LABEL_PATH, device)

    train_sampler = EpisodeBatchSampler(SAMPLE_LABEL_PATH, batch_size=1, model=model, random=False)
    train_dataloader = DataLoader(dataset, batch_sampler=train_sampler, num_workers=0)

    valid_sampler = EpisodeBatchSampler(SAMPLE_LABEL_PATH, batch_size=1, model=model, train=False)
    valid_dataloader = DataLoader(dataset, batch_sampler=valid_sampler, num_workers=0)

    # Define loss function and optimizer
    criterion = nn.MSELoss()

    fig, ax = plt.subplots()
    ax.set(xlabel='Epoch', ylabel='Loss', title='Loss Curve for epochs')
    plt.yscale('log')
    with tqdm(total=EPOCHS_TO_TRAIN * (len(train_dataloader) + len(valid_dataloader))) as pbar:
        for epoch in tqdm(range(EPOCHS_TO_TRAIN), leave=False):
            pbar.write(f'Current Learning Rate: {scheduler.get_last_lr()})')
            display_epoch = starting_epoch + epoch + 1
            model.train()  # switch back to training mode
            sample = 0
            for i, data in tqdm(enumerate(train_dataloader, 0), leave=False):
                inputs, positions, labels = data

                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(inputs, positions)

                # Calculate loss
                loss = calculate_loss(criterion, outputs, labels)

                # Backward pass and optimize
                loss.backward(retain_graph=True)
                optimizer.step()

                batch_losses.append(loss.item())

                sample += labels.shape[0]

                pbar.set_description('[%d, %5d] loss: %.8f' %
                                     (display_epoch, sample, np.mean(batch_losses[-200:])))
                pbar.update(1)
            pbar.write(f'LOSS epoch {display_epoch}: {np.mean(batch_losses)}')

            train_sampler.shuffle_episodes()
            epoch_losses.append(np.mean(batch_losses))
            batch_losses = []  # Reset batch losses

            pbar.write(f'STARTING VALIDATION')

            # Validation
            sample = 0
            model.eval()  # switch to evaluation mode
            with torch.no_grad():
                val_losses = []
                for i, val_data in tqdm(enumerate(valid_dataloader, 0), leave=False):
                    val_inputs, val_positions, val_labels = val_data
                    val_outputs = model(val_inputs, val_positions)
                    val_loss = calculate_loss(criterion, val_outputs, val_labels)

                    val_losses.append(val_loss.item())

                    sample += val_labels.shape[0]
                    pbar.set_description('[%d, %5d] validation loss: %.8f' %
                                     (display_epoch, sample, np.mean(val_losses[-1])))
                    pbar.update(1)

                validation_loss = np.mean(val_losses)
                validation_losses.append(validation_loss)
                pbar.write(f'Validation Loss epoch {display_epoch}: {validation_loss}')

            torch.save(model.state_dict(), MODEL_PATH + f'_{display_epoch}.pth')

            details = {
                "epoch_losses": epoch_losses,
                "validation_losses": validation_losses,
                "optimizer": optimizer.state_dict()
            }
            torch.save(details, DETAILS_PATH)
            scheduler.step()


    plt.plot(range(1, starting_epoch + EPOCHS_TO_TRAIN + 1), epoch_losses, label='Epoch Loss')
    plt.plot(range(1, starting_epoch + EPOCHS_TO_TRAIN + 1), validation_losses, label='Validation Epoch Loss')
    fig.savefig('loss_figs/loss_' + MODEL_PATH.split('/')[1]  + "_" + MODEL_PATH.split('/')[2] +'.png')
    plt.show(block=True)


if __name__ == "__main__":
    train()
