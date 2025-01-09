import os
import shutil
import random
import numpy as np
from PIL import Image
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from constants import DATA_DIRNAME, TRAIN_IMAGES_DIRNAME, TRAIN_LABELS_DIRNAME, \
    VAL_IMAGES_DIRNAME, VAL_LABELS_DIRNAME, TEST_IMAGES_DIRNAME, TEST_LABELS_DIRNAME


def add_noise(image):
    """
    Adds Gaussian noise to an image.

    Args:
        image (PIL.Image.Image): The original image.

    Returns:
        PIL.Image.Image: The noisy image with Gaussian noise added.
    """
    image_array = np.array(image) / 255.0
    noise = np.random.normal(0, 0.5, image_array.shape)
    noisy_image = np.clip(image_array + noise, 0, 1)
    noisy_image = (noisy_image * 255).astype(np.uint8)
    return Image.fromarray(noisy_image)


def process_dataset(source_dir, images_dir, labels_dir, retained_percentage=1):
    """
    Process a dataset by copying original images to an 'images' folder
    and creating noisy versions in a 'labels' folder.

    Args:
        source_dir (str): Path to the source directory containing images.
        images_dir (str): Path to the directory to save original images.
        labels_dir (str): Path to the directory to save noisy labels.
        retained_percentage (float): Percentage of images to retain in the dataset.

    Returns:
        None
    """
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    all_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    retained_files = random.sample(all_files, int(len(all_files) * retained_percentage))

    for file_name in retained_files:
        file_path = os.path.join(source_dir, file_name)
        with Image.open(file_path) as img:
            # Save original image
            img.save(os.path.join(labels_dir, file_name))

            # Create and save noisy image
            noisy_img = add_noise(img)
            noisy_img.save(os.path.join(images_dir, file_name))


def move_random_images_to_test(val_images_dir, val_labels_dir, test_images_dir, test_labels_dir, num_images=1):
    """
    Selects a random image from the validation set and moves it to the test set.
    The corresponding noisy label is deleted.
    
    Args:
        val_images_dir (str): Path to the directory containing validation images.
        val_labels_dir (str): Path to the directory containing validation labels.
        test_dir (str): Path to the test directory.
        num_images (int): Number of random images to move to the test set.

    Returns:
        None
    """
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)
    val_image_files = os.listdir(val_images_dir)

    if val_image_files:
        random_images = random.sample(val_image_files, num_images)
        for image_name in random_images:
            image_path = os.path.join(val_images_dir, image_name)
            label_path = os.path.join(val_labels_dir, image_name)

            # Move the image to the test directory
            shutil.move(image_path, os.path.join(test_images_dir, image_name))

            # Move the label to the test directory
            shutil.move(label_path, os.path.join(test_labels_dir, image_name))

        print(f"Moved {num_images} random images to the test directory.")


def data_processing():
    """
    Process the AFHQ dataset for denoising.
    The dataset is split into train, validation, and test sets.
    """
    # Define directories
    base_dir = os.path.join(DATA_DIRNAME, "afhq/")
    train_source_dir = os.path.join(base_dir, "train/")
    val_source_dir = os.path.join(base_dir, "val/")

    # Process train set
    print("Processing train dataset...")
    for animal in os.listdir(train_source_dir):
        animal_dir = os.path.join(train_source_dir, animal)
        process_dataset(animal_dir, TRAIN_IMAGES_DIRNAME, TRAIN_LABELS_DIRNAME, retained_percentage=0.1)

    # Process val set
    print("Processing validation dataset...")
    for animal in os.listdir(val_source_dir):
        animal_dir = os.path.join(val_source_dir, animal)
        process_dataset(animal_dir, VAL_IMAGES_DIRNAME, VAL_LABELS_DIRNAME, retained_percentage=0.1)

    # Move a random image from val to test
    print("Selecting a random image for the test dataset...")
    move_random_images_to_test(VAL_IMAGES_DIRNAME, VAL_LABELS_DIRNAME, TEST_IMAGES_DIRNAME, TEST_LABELS_DIRNAME, num_images=50)





