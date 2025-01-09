import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from constants import DATA_DIRNAME

def download_dataset(
        kaggle_dataset_handle: str = "andrewmvd/animal-faces",
        target_dir: str = DATA_DIRNAME
        ) -> str:
    """
    Download a Kaggle dataset to the specified target directory.

    Args:
        kaggle_dataset_handle (str): Kaggle dataset handle, e.g., 'minseokkim/animal10n-dataset'.
        target_dir (str): Path to the directory where the dataset should be saved.

    Returns:
        str: Path to the directory where the dataset was downloaded.
    """
    try:
        # Ensure target directory exists
        os.makedirs(target_dir, exist_ok=True)

        # Use Kaggle API to download the dataset
        command = [
            "kaggle", "datasets", "download", kaggle_dataset_handle,
            "-p", target_dir, "--unzip"
        ]
        subprocess.run(command, check=True)

        print(f"Dataset downloaded to: {target_dir}")
        return target_dir
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        return ""
