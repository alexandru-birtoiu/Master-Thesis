import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import get_image_path
import math

from config import ACTIVE_CAMERAS, DETAILS_PATH, LEARNING_RATE, MODEL_PATH, BATCH_SIZE, \
    IMAGE_SIZE, STARTING_EPOCH, TRAIN_MODEL_MORE, DEVICE, USE_LSTM, SAMPLE_LABEL_PATH, EPOCHS_TO_TRAIN, \
    LSTM_LAYERS, EPOCH_LOSSES_PATH, VALIDATION_LOSSES_PATH

class ImageLayer(nn.Module):
    def __init__(self):
        super(ImageLayer, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=(3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=2, padding=1)
        self.conv6 = nn.Conv2d(192, 256, kernel_size=(3, 3), stride=2, padding=1)
        # self.conv7 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=2, padding=1)
        # self.conv8 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=2, padding=1)


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        # x = torch.relu(self.conv7(x))
        # x = torch.relu(self.conv8(x))

        batch_size, height, width, channels = x.size()
        x = x.view(batch_size, channels * height * width)  # Reshape for LSTM
        return x

class Network(nn.Module):
    def __init__(self, no_cameras, device):
        super(Network, self).__init__()
        self.device = device
        self.no_cameras = no_cameras
        
        self.conv_layers = nn.ModuleList([ImageLayer().to(device) for _ in range(no_cameras)])

        conv_output_size = 1024 * no_cameras + 7
        ff_hidden_size = 256 * no_cameras

        if USE_LSTM:
            self.lstm = nn.LSTM(input_size=conv_output_size, hidden_size=ff_hidden_size, num_layers=LSTM_LAYERS, batch_first=True)
        self.fc1 = nn.Linear(conv_output_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, 15)  # 7 velocities + 2 gripper action + 3 cube positions + 3 end effector positions

    def forward(self, x, positions):
        output = torch.tensor([], device=self.device)
        for idx, input in enumerate(x):
            input = self.conv_layers[idx](input)
            output = torch.cat((output, input), dim = 1)

        x = torch.concat((output, positions), dim=1)

        if USE_LSTM: 
            lstm_out, _ = self.lstm(x)
            return self.fc2(lstm_out)
        
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
class EpisodeBatchSampler(Sampler):
    def __init__(self, label_file, batch_size, split=0.85, train=True, random=False):
        self.labels = torch.load(label_file)
        self.batch_size = batch_size
        self.random = random
        
        # keys_to_keep = list(self.labels.keys())[:5000]
        # self.labels = {k: self.labels[k] for k in keys_to_keep}

        self.episode_indices = {}
        for i, item in self.labels.items():
            episode = item['episode']  
            if episode not in self.episode_indices:
                self.episode_indices[episode] = []
            self.episode_indices[episode].append(i)

        self.episodes = list(self.episode_indices.keys())
        split_episodes = int(len(self.episodes) * split)
        if train:
            self.episodes = self.episodes[:split_episodes]
        else:
            self.episodes = self.episodes[split_episodes:]

        self.length = 0
        b = 1
        for episode in self.episodes:
            indices = self.episode_indices[episode]
            for idx in indices:
                b += 1
                if b == self.batch_size:
                    self.length += 1
                    b = 0
            if b > 0:
                self.length += 1
        
        self.shuffle_episodes()

    def shuffle_episodes(self):
        print(len(self.episodes))
        self.episodes = np.random.permutation(self.episodes)

    def __iter__(self):
        batch = []
        for episode in self.episodes:
            indices = self.episode_indices[episode]
            if self.random:
                indices = np.random.permutation(indices)
            for idx in indices:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0:
                yield batch

    def __len__(self):
        return self.length


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
            sample_images = []
            for i in range(3, -1, -1):
                image_path = get_image_path(camera, episode, sample - i)
                img = read_image(image_path).to(self.device, dtype=torch.float)
                sample_images.append(img)
            inputs.append(torch.cat(sample_images, dim=0))
        
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    if TRAIN_MODEL_MORE:
        model.load_state_dict(torch.load(MODEL_PATH + '_' + str(STARTING_EPOCH) + '.pth', map_location=DEVICE))
        model.train()

        details = torch.load(DETAILS_PATH)
        
        starting_epoch = STARTING_EPOCH
        epoch_losses = details["epoch_losses"]
        validation_losses = details["validation_losses"]

        [scheduler.step() for _ in range(starting_epoch)]

    dataset = CustomImageDataset(SAMPLE_LABEL_PATH, device)

    train_sampler = EpisodeBatchSampler(SAMPLE_LABEL_PATH, batch_size=BATCH_SIZE, random=False)
    train_dataloader = DataLoader(dataset, batch_sampler=train_sampler, num_workers=0)

    valid_sampler = EpisodeBatchSampler(SAMPLE_LABEL_PATH, batch_size=BATCH_SIZE, train=False)
    valid_dataloader = DataLoader(dataset, batch_sampler=valid_sampler, num_workers=0)
    print(len(dataset))

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
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

                sample += labels.shape[0]
                pbar.set_description('[%d, %5d] loss: %.8f' %
                                     (display_epoch, sample, np.mean(batch_losses[-1])))
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
    fig.savefig('loss_figs/loss_' + MODEL_PATH.split('/')[1] +'.png')
    plt.show(block=True)


if __name__ == "__main__":
    train()
