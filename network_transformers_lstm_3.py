import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import get_image_path, get_depth_path
import math
from collections import deque
from torchtest import assert_vars_change
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
    


class LayerNormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LayerNormLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, states):
        output, states = self.lstm(x, states)
        output = self.layer_norm(output)
        return output, states

class Network(nn.Module):
    def __init__(self, no_cameras, device):
        super(Network, self).__init__()
        self.device = device
        self.no_cameras = no_cameras
        
        self.conv_layers = nn.ModuleList([ImageLayer(SEQUENCE_LENGTH * NETWORK_IMAGE_LAYER_SIZE).to(device) for _ in range(no_cameras)])
        self.cross_view_attention = CrossViewAttention(256, 8, 512).to(device)

        conv_out_size = 256

        layer_positions = conv_out_size + 7
        
        # self.dropout = nn.Dropout(p=0.3) 

        self.mlp = nn.Sequential(
            nn.Linear(layer_positions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.lstm_input_size = 64
        self.lstm_hidden = 64

        self.lstm = LayerNormLSTM(self.lstm_input_size, self.lstm_hidden, LSTM_LAYERS)

        self.last = nn.Linear(self.lstm_hidden, NETWORK_OUTPUT_SIZE)

        # Register hidden states as buffers
        self._initialize_hidden_state_buffers()

        # Initialize hidden states
        self.reset_hidden_states(BATCH_SIZE)

    def _initialize_hidden_state_buffers(self):
        self.register_buffer('h_last', torch.randn(LSTM_LAYERS, BATCH_SIZE, self.lstm_hidden, requires_grad=False), persistent=False)
        self.register_buffer('c_last', torch.randn(LSTM_LAYERS, BATCH_SIZE, self.lstm_hidden, requires_grad=False), persistent=False)

    def reset_hidden_states(self, batch_size):
        self.h_last = torch.randn(LSTM_LAYERS, batch_size, self.lstm_hidden, requires_grad=False).to(self.device)
        self.c_last = torch.randn(LSTM_LAYERS, batch_size, self.lstm_hidden, requires_grad=False).to(self.device)

    def forward(self, x, positions, sequentials):
        image_output = []

        for idx, input in enumerate(x):
            input = self.conv_layers[idx](input)
            image_output.append(input)

        camera_1 = image_output[0]

        if len(image_output) > 1:
            camera_2 = image_output[1]
            cross_attention_out = self.cross_view_attention(camera_1, camera_2)
        else:
            cross_attention_out = camera_1
        
        mlp_in = torch.concat((cross_attention_out, positions.unsqueeze(1)), dim=2)
        mlp_out = self.mlp(mlp_in)

        # Create mask for sequences with length 1
        mask = sequentials

        # Retrieve or initialize hidden states based on the mask
        h_last = self.h_last
        c_last = self.c_last

        if mask.any():
            lstm_input = mlp_out[mask]
            lstm_out, (h_last[:, mask], c_last[:, mask]) = self.lstm(lstm_input, (h_last[:, mask], c_last[:, mask]))
            lstm_out = lstm_out[:, -1, :]
        else:
            lstm_out = torch.tensor([], device=mlp_out.device)

        if (~mask).any():
            non_lstm_out = mlp_out[~mask].squeeze(1)
        else:
            non_lstm_out = torch.tensor([], device=mlp_out.device)

        # Save the whole detached hidden state
        self.h_last = h_last.detach()
        self.c_last = c_last.detach()

        # Concatenate outputs
        final_output = torch.cat([lstm_out, non_lstm_out], dim=0)

        return self.last(final_output)



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
            self.episodes = episodes[:split_idx]
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
                        sample_batch.append((idx, 1))
                    batch.append(sample_batch)
            else:
                half_batch_size = len(batch_episodes) // 2
                sequential_episodes = batch_episodes[:half_batch_size]
                random_episodes = batch_episodes[half_batch_size:]

                seq_episodes_sample_sequences = {
                    episode: list(self.labels[episode].keys())[SEQUENCE_LENGTH - 1:]
                    for episode in sequential_episodes
                }
                random_episodes_sample_sequences = {
                    episode: np.random.permutation(list(self.labels[episode].keys())[SEQUENCE_LENGTH - 1:])
                    for episode in random_episodes
                }

                max_samples_seq = max(len(seq) for seq in seq_episodes_sample_sequences.values())
                max_samples_rand = max(len(rand) for rand in random_episodes_sample_sequences.values())

                max_samples = max(max_samples_seq, max_samples_rand)

                for sample_idx in range(max_samples):
                    sample_batch = []

                    for episode in sequential_episodes:
                        samples = seq_episodes_sample_sequences[episode]
                        if sample_idx < len(samples):
                            sample = samples[sample_idx]
                            idx = self.dataset_indices[(episode, sample)]
                            sample_batch.append((idx, 1))  # 1 for sequential
                        else:
                            sample_batch.append((None, 0))  # Padding for missing samples

                    for episode in random_episodes:
                        samples = random_episodes_sample_sequences[episode]
                        if sample_idx < len(samples):
                            sample = samples[sample_idx]
                            idx = self.dataset_indices[(episode, sample)]
                            sample_batch.append((idx, 0))  # 0 for random
                        else:
                            sample_batch.append((None, 0))  # Padding for missing samples

                    if not all(item[0] is None for item in sample_batch):
                        batch.append(sample_batch)
                
            self.batches.extend(batch)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

class CustomImageDataset(Dataset):
    def __init__(self, labels, device):
        self.labels = labels
        self.device = device
        self.cameras = ACTIVE_CAMERAS

        self.episode_sample_pairs = [
            (episode, sample) for episode, samples in self.labels.items() \
                for sample in samples.keys()
        ]

        self.image_cache = ImageCache(max_size=len(self.cameras) * MAX_SEQUENCE * BATCH_SIZE)

    def __len__(self):
        return len(self.episode_sample_pairs)

    def __getitem__(self, item):
        idx, sequential = item  # Unpack the tuple

        if idx is None:
            # Return a placeholder tensor and padding indicator for None index
            inputs = [torch.zeros((4 * SEQUENCE_LENGTH, IMAGE_SIZE, IMAGE_SIZE), device=self.device) for _ in self.cameras]
            positions = torch.zeros(7, device=self.device)
            label = torch.zeros(24, device=self.device)  # Adjusted size for task one-hot encoding
            padding_indicator = 1  # Indicates this is a padded sample
            return inputs, positions, label, padding_indicator, sequential, None

        episode, sample = self.episode_sample_pairs[idx]
        inputs = []

        for camera in self.cameras:
            sample_images = []
            for i in range(SEQUENCE_LENGTH - 1, -1, -1):
                if USE_DEPTH:
                    depth_path = get_depth_path(camera, episode, sample - i)
                    depth_data_normalized = self.image_cache.get(depth_path)

                    if depth_data_normalized is None:
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

        #TODO: ADD TASK LABEL FOR TRAY POSITION
        task = self.labels[episode][sample]["task"][0]
        task_one_hot = torch.nn.functional.one_hot(torch.tensor(task - 1), num_classes=3).to(self.device)  # Adjust task to zero-based
        label = torch.cat((label[:-8], task_one_hot.float(), label[-8:]), dim=0)

        padding_indicator = 0
        return inputs, positions, label, padding_indicator, sequential, episode

def collate_fn(batch):
    inputs, positions, labels, padding_indicators, sequentials, episodes = zip(*batch)

    inputs = [torch.stack([inputs[b][i] for b in range(len(inputs))], dim=0) for i in range(len(inputs[0]))]

    positions = torch.stack(positions, dim=0)
    labels = torch.stack(labels, dim=0)
    padding_indicators = torch.tensor(padding_indicators, dtype=torch.float32)

    sequentials = torch.tensor(sequentials).to(device=DEVICE) == 1

    return inputs, positions, labels, padding_indicators, episodes, sequentials

def reset_hidden_states_if_needed(model, current_episodes, new_episodes, batch_size):
    filtered_new_episodes = [ep for ep in new_episodes if ep is not None]

    set_current_episodes = set(current_episodes) if current_episodes is not None else set()
    set_new_episodes = set(filtered_new_episodes)
    
    if not set_new_episodes.issubset(set_current_episodes):

        model.reset_hidden_states(batch_size)
        return filtered_new_episodes  # Update to filtered new episodes

    return current_episodes

def train():
    def run_epoch(model, dataloader, criterion, pbar, epoch, is_training=True, optimizer=None):
        model.train() if is_training else model.eval()
        epoch_losses = []
        current_episodes = None
        sample = 0
        
        with torch.set_grad_enabled(is_training):
            for i, data in tqdm(enumerate(dataloader, 0), leave=False):
                inputs, positions, labels, padding_indicators, episodes, sequentials = data
                current_batch_size = labels.shape[0]
                current_episodes = reset_hidden_states_if_needed(model, current_episodes, episodes, current_batch_size)

                if is_training:
                    optimizer.zero_grad()
                
                outputs = model(inputs, positions, sequentials)
                loss = calculate_loss(criterion, outputs, labels, padding_indicators == 0)
                epoch_losses.append(loss.item())
                
                if is_training:
                    loss.backward()
                    optimizer.step()

                sample += current_batch_size
                pbar.set_description('[%d, %5d] %s loss: %.8f batches: %d' %
                                     (epoch, sample, 'train' if is_training else 'validation', np.mean(epoch_losses[-1]), current_batch_size))
                pbar.update(1)

        return np.mean(epoch_losses)

    device = torch.device(DEVICE)
    model = Network(len(ACTIVE_CAMERAS), device).to(device)
    starting_epoch = 0
    epoch_losses = []
    validation_losses = []

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
    dataset = CustomImageDataset(labels, device)

    train_sampler = EpisodeBatchSampler(dataset.labels, batch_size=BATCH_SIZE, train=True, random=True)
    train_dataloader = DataLoader(dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=collate_fn)

    valid_sampler = EpisodeBatchSampler(dataset.labels, batch_size=BATCH_SIZE, train=False, random=False)
    valid_dataloader = DataLoader(dataset, batch_sampler=valid_sampler, num_workers=0, collate_fn=collate_fn)

    criterion = nn.MSELoss()
    fig, ax = plt.subplots()
    ax.set(xlabel='Epoch', ylabel='Loss', title='Loss Curve for epochs')
    plt.yscale('log')

    with tqdm(total=EPOCHS_TO_TRAIN * (len(train_dataloader) + len(valid_dataloader))) as pbar:
        for epoch in tqdm(range(EPOCHS_TO_TRAIN), leave=False):
            pbar.write(f'Current Learning Rate: {scheduler.get_last_lr()}')
            display_epoch = starting_epoch + epoch + 1

            train_loss = run_epoch(model, train_dataloader, criterion, pbar, display_epoch, is_training=True, optimizer=optimizer)
            epoch_losses.append(train_loss)
            pbar.write(f'LOSS epoch {display_epoch}: {train_loss}')

            train_sampler.shuffle_episodes()
            pbar.write(f'STARTING VALIDATION')
            val_loss = run_epoch(model, valid_dataloader, criterion, pbar, display_epoch, is_training=False)
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
    if not USE_LSTM or not USE_TRANSFORMERS:
        raise ValueError("This network type should be used only when both use_transformers and use_lstm are true.")
    train()



