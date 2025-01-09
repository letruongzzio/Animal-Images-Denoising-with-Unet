import os

PARENT_DIRNAME = os.path.expanduser("~/Animal-Images-Denoising-with-Unet/")
DATA_DIRNAME = os.path.join(PARENT_DIRNAME, "data/")
MODEL_DIRNAME = os.path.join(PARENT_DIRNAME, "model/")
IMAGE_DIRNAME = os.path.join(PARENT_DIRNAME, "image/")
TRAIN_IMAGES_DIRNAME = os.path.join(DATA_DIRNAME, "afhq/data_processed/train/images")
TRAIN_LABELS_DIRNAME = os.path.join(DATA_DIRNAME, "afhq/data_processed/train/labels")
VAL_IMAGES_DIRNAME = os.path.join(DATA_DIRNAME, "afhq/data_processed/val/images")
VAL_LABELS_DIRNAME = os.path.join(DATA_DIRNAME, "afhq/data_processed/val/labels")
TEST_IMAGES_DIRNAME = os.path.join(DATA_DIRNAME, "afhq/data_processed/test/images")
TEST_LABELS_DIRNAME = os.path.join(DATA_DIRNAME, "afhq/data_processed/test/labels")
