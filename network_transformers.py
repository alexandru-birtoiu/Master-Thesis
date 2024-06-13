import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import get_image_path, get_depth_path
import math
from collections import OrderedDict
from config import *
from training_utils import *



class ImageLayer(nn.Module):
    def __init__(self, input_channels):
        super(ImageLayer, self).__init__()
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv2d(input_channels, 12, kernel_size=(5, 5), stride=2, padding=2)  # 5x5 kernel, stride=2
        self.conv2 = nn.Conv2d(12, 24, kernel_size=(3, 3), stride=2, padding=1)  # 3x3 kernel, stride=2
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # Max-pooling layer
        self.conv3 = nn.Conv2d(24, 36, kernel_size=(3, 3), stride=1, padding=1)  # 3x3 kernel, stride=1
        self.conv4 = nn.Conv2d(36, 48, kernel_size=(3, 3), stride=2, padding=1)  # 3x3 kernel, stride=2
        self.conv5 = nn.Conv2d(48, 64, kernel_size=(3, 3), stride=2, padding=1)  # 3x3 kernel, stride=2
        self.conv6 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)  # 3x3 kernel, stride=2
        self.conv7 = nn.Conv2d(128, 256, kernel_size=(2, 2), stride=1, padding=0)  # 3x3 kernel, stride=1

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        number_of_images = channels // self.input_channels

        # Reshape input tensor to (batch_size * number_of_images, channels_per_image, height, width)
        x = x.view(batch_size * number_of_images, self.input_channels, height, width)

        # Apply convolutional layers
        x = torch.relu(self.conv1(x))  
        x = torch.relu(self.conv2(x))  
        x = self.maxpool1(x)           
        x = torch.relu(self.conv3(x))  
        x = torch.relu(self.conv4(x))  
        x = torch.relu(self.conv5(x))  
        x = torch.relu(self.conv6(x))  
        x = torch.relu(self.conv7(x))  

        # Reshape the tensor
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

    def forward(self, z1, z2):
        ln_z1 = self.layer_norm(z1)
        ln_z2 = self.layer_norm(z2)

        q1 = self.q_linear1(ln_z1)
        k2 = self.k_linear2(ln_z2)
        v2 = self.v_linear2(ln_z2)

        q2 = self.q_linear2(ln_z2)
        k1 = self.k_linear1(ln_z1)
        v1 = self.v_linear1(ln_z1)

        attn_output1, _ = self.attention(q1, k2, v2)
        attn_output2, _ = self.attention(q2, k1, v1)

        ln_attn_output1 = self.layer_norm(z1 + attn_output1)
        ln_attn_output2 = self.layer_norm(z2 + attn_output2)

        h1 = self.mlp1(ln_attn_output1)
        h2 = self.mlp2(ln_attn_output2)

        combined_output = h1 + h2
        
        
        return combined_output

import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, no_cameras, device):
        super(Network, self).__init__()
        self.device = device
        self.no_cameras = no_cameras

        self.conv_layers = nn.ModuleList([ImageLayer(SEQUENCE_LENGTH * NETWORK_IMAGE_LAYER_SIZE).to(device) for _ in range(no_cameras)])
        
        if no_cameras > 1:
            self.cross_view_attention = nn.ModuleList()
            for i in range(no_cameras):
                for j in range(i + 1, no_cameras):
                    self.cross_view_attention.append(CrossViewAttention(256, 8, 512).to(device))

        hidden_size = 256
        layer_positions = hidden_size #+ 7

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
        # Apply convolutional layers to each camera input
        image_output = [self.conv_layers[idx](input) for idx, input in enumerate(x)]
        image_output = [output.squeeze(1) for output in image_output]

        if self.no_cameras == 1:
            combined_output = image_output[0]
        else:
            # Compute pairwise cross-view attention
            attention_outputs = []
            attention_idx = 0
            for i in range(self.no_cameras):
                for j in range(i + 1, self.no_cameras):
                    attention_outputs.append(self.cross_view_attention[attention_idx](image_output[i], image_output[j]))
                    attention_idx += 1

            # Average all attention outputs
            combined_output = sum(attention_outputs) / len(attention_outputs)

        # output = torch.cat((combined_output, positions), dim=1)
        output = combined_output
        
        return self.mlp(output)


class EpisodeBatchSampler(Sampler):
    def __init__(self, labels, batch_size, split=0.85, train=True, random=False):
        self.labels = labels
        self.batch_size = batch_size
        self.train = train
        self.random = random
        self.dataset_indices = self._create_dataset_indices()

        episodes = list(self.labels.keys())
        split_idx = int(len(episodes) * split)
        
        if self.train:
            self.episodes = episodes[:split_idx][1:]
        else:
            self.episodes = episodes[split_idx:]

        self.shuffle_episodes()

    def shuffle_episodes(self):
        np.random.shuffle(self.episodes)
        self.batches = []
        self.prepare_batches()

    def _create_dataset_indices(self):
        episode_sample_pairs = [
            (episode, sample) for episode, samples in self.labels.items() \
                for sample in list(samples.keys())
        ]
        return {pair: idx for idx, pair in enumerate(episode_sample_pairs)}

    def prepare_batches(self):
        num_batches = math.ceil(len(self.episodes) / self.batch_size)

        for batch_idx in range(num_batches):
            batch_episodes = self.episodes[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
            batch = []
            if self.random == False:
                max_samples = max(len(self.labels[episode]) for episode in batch_episodes)
                
                for sample_idx in range(SEQUENCE_LENGTH - 1, max_samples):
                    sample_batch = []
                    for episode in batch_episodes:
                        samples = list(self.labels[episode].keys())
                        if sample_idx < len(samples):
                            sample = samples[sample_idx]
                            idx = self.dataset_indices[(episode, sample)]
                            sample_batch.append(idx)
                    batch.append(sample_batch)

            else:
                episode_sample_sequences = {episode: np.random.permutation(list(self.labels[episode].keys())[SEQUENCE_LENGTH - 1:]) for episode in batch_episodes}
                max_samples = max(len(episode_sample_sequences[episode]) for episode in batch_episodes)

                for sample_idx in range(max_samples):
                    sample_batch = []
                    for episode in batch_episodes:
                        samples = episode_sample_sequences[episode]
                        if sample_idx < len(samples):
                            sample = samples[sample_idx]
                            idx = self.dataset_indices[(episode, sample)]
                            sample_batch.append(idx)
                    batch.append(sample_batch)
                
            self.batches.extend(batch)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

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
        model_dict = torch.load(MODEL_PATH + '_' + str(STARTING_EPOCH) + '.pth', map_location=DEVICE)
        model.load_state_dict(model_dict)
        model.train()

        details = torch.load(DETAILS_PATH)
        
        starting_epoch = STARTING_EPOCH
        epoch_losses = details["epoch_losses"][:STARTING_EPOCH]
        validation_losses = details["validation_losses"][:STARTING_EPOCH]

        [scheduler.step() for _ in range(starting_epoch)]

    labels = torch.load(SAMPLE_LABEL_PATH)

    labels = prepare_labels(labels)

    dataset = CustomImageDataset(labels)

    # Create DataLoaders for training and validation sets
    train_sampler = EpisodeBatchSampler(dataset.labels, batch_size=BATCH_SIZE, train=True, random=True)
    train_dataloader = DataLoader(dataset, batch_sampler=train_sampler, num_workers=3, pin_memory=True)

    valid_sampler = EpisodeBatchSampler(dataset.labels, batch_size=BATCH_SIZE, train=False, random=False)
    valid_dataloader = DataLoader(dataset, batch_sampler=valid_sampler, num_workers=3, pin_memory=True)

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

                inputs = [input.to(device) for input in inputs]
                positions = positions.to(device)
                labels = labels.to(device)
            
                current_batch_size = labels.shape[0]

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

                sample += current_batch_size

                pbar.set_description('[%d, %5d] loss: %.8f batches: %d' %
                                     (display_epoch, sample, np.mean(batch_losses[-1]), current_batch_size))
                pbar.update(1)
            pbar.write(f'LOSS epoch {display_epoch}: {np.mean(batch_losses)}')

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


                    val_inputs = [input.to(device) for input in val_inputs]
                    val_positions = val_positions.to(device)
                    val_labels = val_labels.to(device)

                    current_batch_size = val_labels.shape[0]

                    val_outputs = model(val_inputs, val_positions)
                    val_loss = calculate_loss(criterion, val_outputs, val_labels)

                    val_losses.append(val_loss.item())

                    sample += current_batch_size
                    pbar.set_description('[%d, %5d] validation loss: %.8f batches: %d' %
                                     (display_epoch, sample, np.mean(val_losses[-1]), current_batch_size))
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
    if USE_LSTM:
        raise ValueError("This network type should be used only when both use_transformers and use_lstm are false.")
    train()
