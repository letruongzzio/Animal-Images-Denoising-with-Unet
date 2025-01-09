# Animal Images Denoising with UNet

This project implements a UNet-based model for denoising animal images, using the AFHQ dataset. The project pipeline includes data preprocessing, model training, evaluation, and visualization of predictions.

## Table of Contents

- [Animal Images Denoising with UNet](#animal-images-denoising-with-unet)
  - [Table of Contents](#table-of-contents)
    - [Project Overview](#project-overview)
    - [Dataset](#dataset)
    - [File Structure](#file-structure)
    - [Usage](#usage)
    - [Extensions](#extensions)
      - [1. Colorization](#1-colorization)
      - [2. Super-Resolution](#2-super-resolution)

---

### Project Overview

The project focuses on denoising animal images by adding Gaussian noise and training a UNet model to reconstruct the clean version. Key features include:
- **Data preprocessing**: Splitting the dataset into train, validation, and test sets, and adding Gaussian noise.
- **Model**: A UNet architecture is employed for image denoising.
- **Visualization**: Input images, predictions, and ground truth are visualized during training and evaluation.

---

### Dataset

The dataset used is [Animal Faces HQ (AFHQ)](https://www.kaggle.com/datasets/andrewmvd/animal-faces), which contains high-quality images of cats, dogs, and wild animals. 

- **Train Set**: Clean and noisy versions of images for training.
- **Validation Set**: Used for model evaluation during training.
- **Test Set**: A small subset moved from validation for final testing.

You should download the dataset and extract it into the `data/` directory before running the code as described in the setup instructions [README](../utils/README.md).

---

### File Structure

```
Animal-Images-Denoising-with-Unet/
├── model/
│   ├── animal_dataset.py
│   ├── data_loader.py
│   ├── unet_model.py
│   ├── modelPipeline.py
│   ├── storage/
│       ├── train_loader.pth
│       ├── val_loader.pth
│       ├── test_loader.pth
│       ├── trained_unet.pth
```

---

### Usage

1. Training the Model:
   ```bash
   python3 model/modelPipeline.py
   ```

2. Testing the Model:
   ```bash
   python3 test.py
   ```

---

### Extensions

This project can be extended for **Colorization** and **Super-Resolution** tasks by making the following changes:

#### 1. Colorization
- **Input**: Convert grayscale images to input tensors.
- **Output**: Modify the model to generate the colorized RGB channels.
- **Changes in `animal_dataset.py`**: Convert grayscale images for input using `image.convert('L')` during preprocessing.
- **Changes in `unet_model.py`**: Ensure the final output layer produces 3 channels (RGB).

#### 2. Super-Resolution
- **Input**: Downscale the original images for the input.
- **Output**: Modify the model to generate higher-resolution images.
- **Changes in `animal_dataset.py`**: Resize images to lower resolution using `transforms.Resize((low_height, low_width))`.
- **Changes in `unet_model.py`**:
  - Adjust the model’s architecture to upsample inputs to the desired resolution.
  - Use `ConvTranspose2D` or `PixelShuffle` for upsampling.
