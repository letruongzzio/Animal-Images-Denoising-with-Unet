import torch
import copy
import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet_model import UNet
from data_loader import get_dataloaders
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from constants import MODEL_DIRNAME, IMAGE_DIRNAME


def evaluate(model, loader, criterion, device, val_display=True, display_sample=True):
    """
    Evaluate the model on the given DataLoader and optionally display a random sample.

    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): The DataLoader for evaluation.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): The device to run the model on.
        mean (list): Mean used for normalization.
        std (list): Std used for normalization.
        display_sample (bool): Whether to display a random sample.

    Returns:
        float: The average loss over the dataset.
    """
    model.eval()
    total_loss = 0.0
    random_sample = None

    with torch.no_grad():
        for _, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Save a random sample
            if display_sample and random_sample is None:
                random_idx = random.randint(0, len(inputs) - 1)
                random_sample = (inputs[random_idx], labels[random_idx])

    # Display the random sample
    if val_display and display_sample and random_sample:
        input_img, target_img = random_sample
        display_prediction(
            model=model,
            image=input_img,
            target=target_img,
            device=device,
            title="val_prediction"
        )

    return total_loss / len(loader)



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training loop with tqdm progress bar
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            torch.cuda.empty_cache()

        # Calculate average losses
        train_loss = running_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, criterion, device)

        # Save best model weights
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        # Append losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Display progress
        print(f"\tRESULT: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Restore best model weights
    model.load_state_dict(best_model_wts)
    # Save model
    torch.save(model.state_dict(), os.path.join(MODEL_DIRNAME, "storage/trained_unet.pth"))
    return train_losses, val_losses, model


def denormalize(image, mean, std, device):
    """
    Denormalize a tensor image using the given mean and std.

    Args:
        image (torch.Tensor): Normalized tensor image of shape (C, H, W).
        mean (list): Mean used for normalization (one value per channel).
        std (list): Std used for normalization (one value per channel).
        device (torch.device): The device on which the tensors are located.

    Returns:
        torch.Tensor: Denormalized image tensor.
    """
    # Move mean and std to the same device as the image
    mean = torch.tensor(mean, device=device).reshape(-1, 1, 1)
    std = torch.tensor(std, device=device).reshape(-1, 1, 1)
    image = image.to(device) * std + mean
    return image


# Updated display_prediction function
def display_prediction(
        model,
        image,
        target,
        device,
        title,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ):
    """
    Display input, prediction, and ground truth for a given image and model.

    Args:
        model (torch.nn.Module): The trained model.
        image (torch.Tensor): The input noisy image.
        target (torch.Tensor): The clean ground truth image.
        device (torch.device): The device to run the model on.
        mean (list): Mean used for normalization.
        std (list): Std used for normalization.
    """
    model.eval()

    # Ensure the input image is 4D (add batch dimension if needed)
    if len(image.shape) == 3:  # Shape: [C, H, W]
        img = image.unsqueeze(0).to(device)  # Add batch dimension
    else:  # Shape: [B, C, H, W]
        img = image.to(device)

    # Get model output
    output = model(img)[0].detach().cpu()  # Remove batch dimension

    # Denormalize images
    input_img = denormalize(image.squeeze(0), mean, std, device).permute(1, 2, 0).clip(0, 1).cpu().numpy()
    pred_img = denormalize(output, mean, std, device).permute(1, 2, 0).clip(0, 1).cpu().numpy()
    target_img = denormalize(target.squeeze(0), mean, std, device).permute(1, 2, 0).clip(0, 1).cpu().numpy()

    # Add index to title if multiple images with the same title
    existing_titles = [f for f in os.listdir(IMAGE_DIRNAME) if f.startswith(title)]
    title = f"{title}_{len(existing_titles) + 1}"

    # Display images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title("Input Image")
    plt.imshow(input_img)

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title("Prediction")
    plt.imshow(pred_img)

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title("Ground Truth")
    plt.imshow(target_img)

    plt.savefig(os.path.join(IMAGE_DIRNAME, f"{title}.png"))
    plt.show()


# Plotting function for losses
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(IMAGE_DIRNAME, "loss_plot.png"))
    plt.show()


def model_pipeline(getDataLoader=True):
    # Define the model parameters
    max_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = UNet(in_channels=3, out_channels=3)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load the data
    print("Loading the data...")
    if getDataLoader:
        get_dataloaders(batch_size=[2, 4, 16])
    train_loader = torch.load(os.path.join(MODEL_DIRNAME, "storage/train_loader.pth"))
    val_loader = torch.load(os.path.join(MODEL_DIRNAME, "storage/val_loader.pth"))

    # Train the model
    print("Training the model...")
    train_losses, val_losses, model = train_model(model, train_loader, val_loader, criterion, optimizer, max_epochs, device)

    # Plot the losses
    print("Plotting the training and validation losses...")
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    model_pipeline()
    torch.cuda.empty_cache()

