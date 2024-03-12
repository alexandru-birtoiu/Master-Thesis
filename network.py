import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import MODEL_PATH, BATCH_SIZE, EPOCH, STARTING_EPOCH, TRAIN_MODEL_MORE, DEVICE, IMAGES_PATH


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=(3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=2, padding=1)
        self.conv6 = nn.Conv2d(192, 256, kernel_size=(3, 3), stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=2, padding=1)
        # self.conv8 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=2, padding=1)
        self.lstm = nn.LSTM(input_size=1031, hidden_size=128, num_layers=1, batch_first=True)
        # self.fc1 = nn.Linear(1031, 256)
        self.fc2 = nn.Linear(128, 15)  # 7 velocities + 2 gripper action + 3 cube positions + 3 end effector positions

    def forward(self, x, positions):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.relu(self.conv7(x))
        # x = torch.relu(self.conv8(x))

        batch_size, height, width, channels = x.size()
        x = x.view(batch_size, channels * height * width)  # Reshape for LSTM

        x = torch.concat((x, positions), dim=1)

        lstm_out, _ = self.lstm(x)

        # x = torch.relu(self.fc1(x))
        # return self.fc2(x)
        return self.fc2(lstm_out)  # Taking the last time step output of LSTM


class CustomImageDataset(Dataset):
    def __init__(self, label_file, image_path, device):
        self.labels = torch.load(label_file)
        self.image_path = image_path
        self.device = device

    def __len__(self):
        return len(self.labels.keys())

    def __getitem__(self, idx):
        sample_images = []
        for i in range(4):
            image_path = self.image_path + str(idx) + '_' + str(i) + '.png'
            img = read_image(image_path).to(self.device, dtype=torch.float)
            sample_images.append(img)
        inputs = torch.cat(sample_images, dim=0)
        label = torch.tensor(self.labels[idx]["labels"]).to(self.device)
        positions = torch.tensor(self.labels[idx]["positions"]).to(self.device)
        return inputs, positions, label


def calculate_loss(criterion, outputs, labels):
    loss_1 = criterion(outputs[:, :7], labels[:, :7])
    loss_2 = criterion(outputs[:, 7:9], labels[:, 7:9])
    loss_3 = criterion(outputs[:, 9:12], labels[:, 9:12])
    loss_4 = criterion(outputs[:, 12:15], labels[:, 12:15])
    return loss_1 + loss_2 + loss_3 + loss_4


def train():
    device = torch.device(DEVICE)

    model = Network().to(device)

    starting_epoch = 0

    if TRAIN_MODEL_MORE:
        model.load_state_dict(torch.load(MODEL_PATH + '_' + str(STARTING_EPOCH) + '.pth', map_location=DEVICE))
        model.train()
        starting_epoch = STARTING_EPOCH

    dataset = CustomImageDataset('sample_labels', IMAGES_PATH, device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(len(dataset))

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 50

    epoch_losses = []
    batch_losses = []

    fig, ax = plt.subplots()
    ax.set(xlabel='Epoch', ylabel='Loss', title='Loss Curve for epochs')
    plt.yscale('log')
    with tqdm(total=num_epochs * len(dataloader)) as pbar:
        for epoch in tqdm(range(num_epochs), leave=False):
            for i, data in tqdm(enumerate(dataloader, 0), leave=False):
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

                pbar.set_description('[%d, %5d] loss: %.8f' %
                                     (starting_epoch + epoch + 1, i * BATCH_SIZE + 1, np.mean(batch_losses[-10:])))
                pbar.update(1)

            epoch_losses.append(np.mean(batch_losses))
            batch_losses = []  # Reset batch losses
            torch.save(model.state_dict(), MODEL_PATH + f'_{starting_epoch + epoch + 1}.pth')
            torch.save(epoch_losses, MODEL_PATH + 'epoch_losses')

    plt.plot(range(1, num_epochs + 1), epoch_losses, label='Epoch Loss')
    fig.savefig('loss_256_50k-lstm.png')
    plt.show(block=True)


if __name__ == "__main__":
    train()
