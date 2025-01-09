from prepare import download_dataset
from processing import data_processing
import torch

def data_pipeline():
    """
    Download and process the dataset.
    """
    download_dataset()
    data_processing()
    print("All datasets processed successfully!")

if __name__ == "__main__":
    data_pipeline()
    torch.cuda.empty_cache()
