import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from animal_dataset import AnimalDataset
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from constants import MODEL_DIRNAME, TRAIN_IMAGES_DIRNAME, TRAIN_LABELS_DIRNAME, \
    VAL_IMAGES_DIRNAME, VAL_LABELS_DIRNAME, TEST_IMAGES_DIRNAME, TEST_LABELS_DIRNAME


IMAGE_SIZE = 224


def get_dataloaders(
        train_images_dir=TRAIN_IMAGES_DIRNAME,
        train_labels_dir=TRAIN_LABELS_DIRNAME,
        val_images_dir=VAL_IMAGES_DIRNAME,
        val_labels_dir=VAL_LABELS_DIRNAME,
        test_image_dir=TEST_IMAGES_DIRNAME,
        test_labels_dir=TEST_LABELS_DIRNAME,
        batch_size=None,
        image_size=IMAGE_SIZE):
    """
    Creates and returns DataLoaders for train, validation, and test datasets.

    Args:
        train_images_dir (str): Path to the directory containing training images.
        train_labels_dir (str): Path to the directory containing training labels.
        val_images_dir (str): Path to the directory containing validation images.
        val_labels_dir (str): Path to the directory containing validation labels.
        test_image_dir (str): Path to the directory containing test images.
        batch_size (list): List containing batch sizes for train, val, and test DataLoaders.
        image_size (int): Size of the images (assumed to be square

    Returns:
        None
    """

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Initialize datasets
    train_dataset = AnimalDataset(image_dir=train_images_dir, label_dir=train_labels_dir, transform=transform)
    val_dataset = AnimalDataset(image_dir=val_images_dir, label_dir=val_labels_dir, transform=transform)
    test_dataset = AnimalDataset(image_dir=test_image_dir, label_dir=test_labels_dir, transform=transform)

    # Initialize DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size[2], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size[1], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size[0], shuffle=False)

    # Save DataLoaders
    torch.save(train_loader, os.path.join(MODEL_DIRNAME, "storage/train_loader.pth"))
    torch.save(val_loader, os.path.join(MODEL_DIRNAME, "storage/val_loader.pth"))
    torch.save(test_loader, os.path.join(MODEL_DIRNAME, "storage/test_loader.pth"))

    torch.cuda.empty_cache()

