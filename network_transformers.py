import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import get_image_path, get_depth_path
import math
from collections import OrderedDict
from torchtest import assert_vars_change
from config import *


class ImageLayer(nn.Module):
    def __init__(self, input_channels):
        super(ImageLayer, self).__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=(3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=2, padding=1)
        self.conv6 = nn.Conv2d(192, 256, kernel_size=(3, 3), stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=2, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        number_of_images = channels // self.input_channels

        # Reshape input tensor to (batch_size * number_of_images, channels_per_image, height, width)
        x = x.view(batch_size * number_of_images, self.input_channels, height, width)

        # Apply convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.relu(self.conv7(x))

        # Reshape back to (batch_size, number_of_images, -1)
        x = x.view(batch_size, number_of_images, -1)

        return x
    
class CrossViewAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim):
        super(CrossViewAttention, self).__init__()
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self.q_linear1 = nn.Linear(embed_dim, embed_dim)
        self.k_linear2 = nn.Linear(embed_dim, embed_dim)
        self.v_linear2 = nn.Linear(embed_dim, embed_dim)
        
        self.q_linear2 = nn.Linear(embed_dim, embed_dim)
        self.k_linear1 = nn.Linear(embed_dim, embed_dim)
        self.v_linear1 = nn.Linear(embed_dim, embed_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        
        self.mlp1 = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )

    def forward(self, camera_1, camera_2):
        _, num_images, _ = camera_1.size()
        
        # Prepare the tensors for MultiheadAttention
        camera_1 = camera_1.permute(1, 0, 2)  # (num_images, batch_size, num_features)
        camera_2 = camera_2.permute(1, 0, 2)  # (num_images, batch_size, num_features)
        
        combined_outputs = []
        
        for i in range(num_images):
            z1 = camera_1[i]
            z2 = camera_2[i]
            
            ln_z1 = self.layer_norm(z1)
            ln_z2 = self.layer_norm(z2)
            
            q1 = self.q_linear1(ln_z1)
            k2 = self.k_linear2(ln_z2)
            v2 = self.v_linear2(ln_z2)
            
            q2 = self.q_linear2(ln_z2)
            k1 = self.k_linear1(ln_z1)
            v1 = self.v_linear1(ln_z1)
            
            attn_output1, _ = self.attention(q1.unsqueeze(0), k2.unsqueeze(0), v2.unsqueeze(0))
            attn_output2, _ = self.attention(q2.unsqueeze(0), k1.unsqueeze(0), v1.unsqueeze(0))
            
            # Squeeze the first dimension
            attn_output1 = attn_output1.squeeze(0)
            attn_output2 = attn_output2.squeeze(0)
            
            ln_attn_output1 = self.layer_norm(z1 + attn_output1)
            ln_attn_output2 = self.layer_norm(z2 + attn_output2)
            
            h1 = self.mlp1(ln_attn_output1)
            h2 = self.mlp2(ln_attn_output2)
            
            combined_output = h1 + h2
            combined_outputs.append(combined_output)
        
        combined_outputs = torch.stack(combined_outputs, dim=0)
        combined_outputs = combined_outputs.permute(1, 0, 2)  # (batch_size, num_images, num_features)
        
        return combined_outputs

class Network(nn.Module):
    def __init__(self, no_cameras, device):
        super(Network, self).__init__()
        self.device = device
        self.no_cameras = no_cameras
        
        self.conv_layers = nn.ModuleList([ImageLayer(1 if USE_DEPTH else 4).to(device) for _ in range(no_cameras)])
        self.cross_view_attention = CrossViewAttention(256, 8, 512).to(device)

        lstm_input_size = 256
        hidden_size1 = 512
        self.hidden_size2 = 128

        last_layer_size = self.hidden_size2 + 7
        
        # self.lstm_camera1 = nn.LSTM(input_size=lstm_input_size, hidden_size=self.hidden_size2, num_layers=LSTM_LAYERS, batch_first=True)
        # self.lstm_camera2 = nn.LSTM(input_size=lstm_input_size, hidden_size=self.hidden_size2, num_layers=LSTM_LAYERS, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(lstm_input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, self.hidden_size2),
            nn.ReLU()
        )
        self.last = nn.Linear(last_layer_size, 11 if PREDICTION_TYPE == PredictionType.POSITION else 15)  
        # 3 next position / 7 joint velocities + 2 gripper action + 3 cube positions + 3 end effector positions

    # def initialize_lstm_weights(self):
    #     for lstm in [self.lstm_camera1, self.lstm_camera2]:
    #         for name, param in lstm.named_parameters():
    #             if 'weight' in name:
    #                 nn.init.xavier_normal_(param)
    #             elif 'bias' in name:
    #                 nn.init.constant_(param, 0)

    def forward(self, x, positions):
        image_output = []
        for idx, input in enumerate(x):
            input = self.conv_layers[idx](input)
            image_output.append(input)

        camera_1 = image_output[0]
        camera_2 = image_output[1]

        # # Forward pass through LSTM layers separately
        # h_0 = torch.randn(LSTM_LAYERS, camera_1.shape[0], self.hidden_size2, requires_grad=False).to(self.device)
        # c_0 = torch.randn(LSTM_LAYERS, camera_1.shape[0], self.hidden_size2, requires_grad=False).to(self.device)
    
        # lstm_out_1, _ = self.lstm_camera1(camera_1, (h_0, c_0))
        # lstm_out_1 = lstm_out_1[:, -1, :]

        # lstm_out_2, _ = self.lstm_camera2(camera_2, (h_0, c_0))
        # lstm_out_2 = lstm_out_2[:, -1, :]

        # cross_attention_out = self.cross_view_attention(lstm_out_1, lstm_out_2)


        cross_attention_out = self.cross_view_attention(camera_1, camera_2)
        lstm_input = cross_attention_out.squeeze(1)
        cross_attention_out = self.mlp(lstm_input)

        
        output = torch.concat((cross_attention_out, positions), dim=1)
        return self.last(output)
    
class EpisodeBatchSampler(Sampler):
    def __init__(self, label_file, batch_size, split=0.85, train=True, random=False):
        self.labels = torch.load(label_file)
        self.batch_size = batch_size
        self.train = train
        self.random = random

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
            (episode, sample) for episode, samples in self.labels.items() \
                for sample in list(samples.keys())[SEQUENCE_LENGTH - 1:]
        ]
        return {pair: idx for idx, pair in enumerate(episode_sample_pairs)}

    def __iter__(self):
        batch = []
        for episode in self.episodes:
            samples = list(self.labels[episode].keys())[SEQUENCE_LENGTH - 1:]
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

    def __len__(self):
        total_samples = sum(math.ceil(len(list(self.labels[episode].keys())[SEQUENCE_LENGTH - 1:]) / self.batch_size) \
                             for episode in self.episodes)
        return total_samples

class ImageCache:
    def __init__(self, max_size):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, path):
        if path in self.cache:
            return self.cache[path]
        return None

    def put(self, path, tensor):
        self.cache[path] = tensor
        self.cache.move_to_end(path)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
            

class CustomImageDataset(Dataset):
    def __init__(self, label_file, device):
        self.labels = torch.load(label_file)
        self.device = device
        self.cameras = ACTIVE_CAMERAS

        self.episode_sample_pairs = [
            (episode, sample) for episode, samples in self.labels.items() \
                for sample in list(samples.keys())[SEQUENCE_LENGTH - 1:]
        ]

        self.image_cache = ImageCache(max_size=len(self.cameras) * 130)

    def __len__(self):
        return len(self.episode_sample_pairs)

    def __getitem__(self, idx):
        episode, sample = self.episode_sample_pairs[idx]
        inputs = []

        for camera in self.cameras:
            sample_images = []
            for i in range(SEQUENCE_LENGTH - 1, -1, -1):
                if USE_DEPTH:
                    depth_path = get_depth_path(camera, episode, sample - i)
                    depth_data_normalized = self.image_cache.get(depth_path)

                    if depth_data is None:
                        depth_data = torch.tensor(torch.load(depth_path))
                        depth_data = depth_data.view(1, IMAGE_SIZE, IMAGE_SIZE)

                        min_val = depth_data.min()
                        max_val = depth_data.max()
                        depth_data_normalized = (depth_data - min_val) / (max_val - min_val)

                        self.image_cache.put(depth_path, depth_data_normalized.to(self.device))

                    sample_images.append(depth_data_normalized)
                else:
                    image_path = get_image_path(camera, episode, sample - i)
                    img = self.image_cache.get(image_path)

                    if img is None:
                        img = read_image(image_path).to(self.device, dtype=torch.float)
                        self.image_cache.put(image_path, img)

                    sample_images.append(img)
            inputs.append(torch.concat(sample_images, dim=0))
        
        label = torch.tensor(self.labels[episode][sample]["labels"]).to(self.device)
        positions = torch.tensor(self.labels[episode][sample]["positions"]).to(self.device)
        return inputs, positions, label


def calculate_loss(criterion, outputs, labels):
    if PREDICTION_TYPE == PredictionType.POSITION:
        loss_1 = criterion(outputs[:, :3], labels[:, 7:10])
    else:
        loss_1 = criterion(outputs[:, :7], labels[:, :7])

    loss_2 = criterion(outputs[:, -8:-6], labels[:, -8:-6])
    loss_3 = criterion(outputs[:, -6:-3], labels[:, -6:-3])
    loss_4 = criterion(outputs[:, -3:], labels[:, -3:])

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

    print(MODEL_PATH)

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

                # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRADIENTS_VALUE)

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
    fig.savefig('loss_figs/loss_' + MODEL_PATH.split('/')[1]  + "_" + MODEL_PATH.split('/')[2] +'.png')
    plt.show(block=True)

if __name__ == "__main__":
    train()
