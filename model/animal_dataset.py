import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class AnimalDataset(Dataset):
    """
    A PyTorch Dataset class for managing and loading Animal Denoising datasets.

    This class is designed to handle image denoising tasks, where the inputs are noisy images
    and the targets are clean images.

    Attributes:
        image_dir (str): Directory containing noisy images (inputs).
        label_dir (str): Directory containing clean images (targets).
        transform (callable): Optional transform to be applied on a sample.
    """

    def __init__(self, image_dir, label_dir, transform=None):
        """
        Initialize the AnimalDataset.

        Args:
            image_dir (str): Directory containing noisy images (inputs).
            label_dir (str): Directory containing clean images (targets).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_filenames = os.listdir(image_dir)
        self.label_filenames = os.listdir(label_dir)
        self.transform = transform

        # Ensure the number of images and labels match
        assert len(self.image_filenames) == len(self.label_filenames), \
            "Number of noisy images and clean images must match."

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary with keys 'noisy' and 'clean', containing the noisy
                  image and the clean image respectively.
        """
        # Load noisy image
        noisy_image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        noisy_image = Image.open(noisy_image_path).convert("RGB")

        # Load clean image
        clean_image_path = os.path.join(self.label_dir, self.label_filenames[idx])
        clean_image = Image.open(clean_image_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        # Convert to tensor (if not done in transform)
        if not isinstance(noisy_image, torch.Tensor):
            noisy_image = torch.from_numpy(np.array(noisy_image).transpose((2, 0, 1))).float() / 255.0
        if not isinstance(clean_image, torch.Tensor):
            clean_image = torch.from_numpy(np.array(clean_image).transpose((2, 0, 1))).float() / 255.0

        return (noisy_image, clean_image)
