import matplotlib.pyplot as plt
from config import *
import torch

def replot_loss():
    # Load the saved details
    details = torch.load(f'{MODEL_PATH}_details')

    # Extract loss details
    epoch_losses = details["epoch_losses"]
    validation_losses = details["validation_losses"]

    # Ensure losses are available
    if not epoch_losses or not validation_losses:
        print("No losses available to plot.")
        return

    # Create a new plot for the losses
    fig, ax = plt.subplots()

    # Plot epoch losses
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')

    # Plot validation losses
    ax.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss')

    # Set log scale for y-axis
    ax.set_yscale('log')

    # Set axis labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss over Epochs')

    # Add legend to the plot
    ax.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    replot_loss()