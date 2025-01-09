# Utils Module Documentation

This module is responsible for downloading, processing, and preparing the dataset for training a denoising U-Net model on animal face images. The steps involve downloading the dataset, adding noise, organizing data into train, validation, and test sets, and ensuring the processed data is ready for use in the pipeline.

---

## **Dataset Information**

- **Dataset Source**: [Animal Faces Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/animal-faces)
- **License**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

---

## **Functions and Scripts**

### **1. `prepare.py`**
Handles dataset downloading from Kaggle.

- **Function**: `download_dataset(kaggle_dataset_handle, target_dir)`
  - Downloads and unzips the dataset into the target directory using the Kaggle API.
  - **Arguments**:
    - `kaggle_dataset_handle`: Kaggle dataset identifier (default: `"andrewmvd/animal-faces"`).
    - `target_dir`: Path to save the dataset.
  - **Example**:
    ```bash
    kaggle datasets download andrewmvd/animal-faces -p ./data --unzip
    ```

### **2. `processing.py`**
Handles dataset preprocessing.

#### **Key Functions**:
1. **`add_noise(image)`:** Adds Gaussian noise to an image.

2. **`process_dataset(source_dir, images_dir, labels_dir, retained_percentage)`:** Copies original images to `images_dir` and creates noisy labels in `labels_dir`.

3. **`move_random_images_to_test(val_images_dir, val_labels_dir, test_images_dir, test_labels_dir, num_images)`:** Moves a random subset of validation images to the test set.

4. **`data_processing()`:** Processes the dataset by:
     1. Splitting the dataset into train, validation, and test sets.
     2. Adding noise to the images.
     3. Ensuring data organization for model training.

### **3. `dataPipeline.py`**
- Combines `prepare.py` and `processing.py` into a single pipeline.

- **Function**: `data_pipeline()`
  - Downloads the dataset using `prepare.py`.
  - Processes the dataset using `processing.py`.
  - **Example**:
    ```bash
    python3 dataPipeline.py
    ```

### **4. `__init__.py`**
Provides imports for easier access to all utility functions.

---

## **Workflow**

1. **Download Dataset**:
   - Ensure Kaggle API is installed and set up.
   - Run the `data_pipeline()` function:
     ```bash
     python3 utils/dataPipeline.py
     ```

2. **Dataset Preprocessing**:
   - Adds Gaussian noise to images.
   - Splits the data into `train`, `val`, and `test` sets.
   - Moves a random subset from validation to test.

3. **Structure After Processing**:
   ```
   DATA_DIRNAME/
   ├── train/
   │   ├── images/ (noisy images)
   │   └── labels/ (original images)
   ├── val/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

---

## **Example Usage**

1. **Run the Pipeline:**
   ```bash
   python3 utils/dataPipeline.py
   ```
2. **View Processed Dataset:**
   Check the structure in the `DATA_DIRNAME` directory.

---

## **Notes**

1. Ensure you have sufficient disk space for downloading and processing the dataset.
2. The script uses `retained_percentage=0.1` to sample 10% of the data for train/ val/ test splits. Adjust as necessary.
3. Test images are randomly selected from the validation set and moved to a separate directory.
