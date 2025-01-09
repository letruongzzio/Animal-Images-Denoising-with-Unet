import sys
import os
PROJECT_ROOT = os.path.expanduser("~/Animal-Images-Denoising-with-Unet/")
sys.path.append(PROJECT_ROOT)

from model.animal_dataset import AnimalDataset
from model.data_loader import get_dataloaders
from model.unet_model import UNet
from model.modelPipeline import train_model, evaluate, display_prediction