import torch
import numpy as np
import nibabel as nib
import pandas as pd


def create_healthy_and_anomalous_list(data_path, output_path):
    """
    Create lists of healthy and anomalous patients and brain slices.

    :param data_path: path to data
    :param output_path: path to save output
    """

    df = pd.read_excel(f"{data_path}/BraTS2023_2017_GLI_Mapping.xlsx")
    brats23_column = df["BraTS2023"]
    all_patients = [element[len("BraTS-GLI-"):] for element in brats23_column if
                    isinstance(element, str) and element.startswith("BraTS-GLI-")]

    brain_slices = np.arange(80, 100)  # between 80 and 100 best; also tried (50, 121)

    healthy_list = []
    anomalous_list = []

    for patient in all_patients:
        path = f"{data_path}/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-{patient}/BraTS-GLI-{patient}"
        seg = nib.load(f"{path}-seg.nii.gz").get_fdata()

        for brain_slice in brain_slices:
            seg_slice = seg[:, :, brain_slice]

            if np.all(seg_slice == 0):  # No tumor
                healthy_list.append([patient, brain_slice])
            else:   # Tumor
                anomalous_list.append([patient, brain_slice])

    print(f"Healthy: {len(healthy_list)}, anomalous: {len(anomalous_list)}")

    healthy_list_path = f"{output_path}/healthy_list.npy"
    anomalous_list_path = f"{output_path}/anomalous_list.npy"

    np.save(healthy_list_path, np.array(healthy_list))
    np.save(anomalous_list_path, np.array(anomalous_list))

    print(f"Healthy list saved as {healthy_list_path}.")
    print(f"Anomalous list saved as {anomalous_list_path}.")


def create_healthy_datasets(data_path, output_path):
    """
    Create training and test datasets of healthy patients and brain slices.

    :param data_path: path to data
    :param output_path: path to save output
    """

    healthy_list = np.load(f"{output_path}/healthy_list.npy")
    healthy_dataset = np.zeros((len(healthy_list), 240, 240), dtype=np.float64)

    for i, (patient, brain_slice) in enumerate(healthy_list):
        path = f"{data_path}/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-{patient}/BraTS-GLI-{patient}"
        healthy_dataset[i] = nib.load(f"{path}-t2f.nii.gz").get_fdata()[:, :, int(brain_slice)]

    # Split into train and test datasets
    np.random.shuffle(healthy_dataset)
    split_idx = int(len(healthy_dataset) * 0.9)
    train_healthy_data = healthy_dataset[:split_idx]
    test_healthy_data = healthy_dataset[split_idx:]
    print(f"Train healthy dataset: {len(train_healthy_data)}, test healthy dataset: {len(test_healthy_data)}")

    # Save datasets
    train_healthy_path = f"{output_path}/train_healthy_dataset.npy"
    test_healthy_path = f"{output_path}/test_healthy_dataset.npy"
    np.save(train_healthy_path, train_healthy_data)
    np.save(test_healthy_path, test_healthy_data)
    print(f"Train healthy datasets saved as {train_healthy_path}.")
    print(f"Test healthy datasets saved as {test_healthy_path}.")


def create_anomalous_dataset(data_path, output_path, num_test_data=1000):
    """
    Create test dataset of anomalous patients and brain slices.

    :param data_path: path to data
    :param output_path: path to save output
    :param num_test_data: number of test data
    """

    anomalous_list = np.load(f"{output_path}/anomalous_list.npy")
    anomalous_dataset = np.zeros((num_test_data, 240, 240), dtype=np.float64)
    anomalous_masks = np.zeros((num_test_data, 240, 240), dtype=np.float64)

    test_anomalous_list = anomalous_list[np.random.choice(anomalous_list.shape[0], num_test_data, replace=False), :]
    for i, (patient, brain_slice) in enumerate(test_anomalous_list):
        path = f"{data_path}/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-{patient}/BraTS-GLI-{patient}"
        anomalous_dataset[i] = nib.load(f"{path}-t2f.nii.gz").get_fdata()[:, :, int(brain_slice)]
        anomalous_masks[i] = nib.load(f"{path}-seg.nii.gz").get_fdata()[:, :, int(brain_slice)]
    print(f"Test anomalous dataset: {len(anomalous_dataset)}.")

    test_dataset_path = f"{output_path}/test_anomalous_dataset.npy"
    test_masks_path = f"{output_path}/test_anomalous_masks.npy"
    np.save(test_dataset_path, anomalous_dataset)
    np.save(test_masks_path, anomalous_masks)
    print(f"Test anomalous dataset and mask saved as {test_dataset_path} and {test_masks_path}.")


def preprocess_datasets(output_path):
    """Preprocess the train and test datasets. [deprecated]"""

    train_dataset_raw = np.load(f"{output_path}/healthy_dataset_raw.npy")
    test_dataset_raw = np.load(f"{output_path}/anomalous_dataset_raw.npy")
    print(f" Train dataset shape: {train_dataset_raw.shape}, test dataset shape: {test_dataset_raw.shape}")

    # raw range -42.56834030151367 to 16421.0
    clip_min = 0
    clip_max = np.percentile(train_dataset_raw, 99)

    train_dataset = np.clip(train_dataset_raw, clip_min, clip_max)
    test_dataset = np.clip(test_dataset_raw, clip_min, clip_max)

    train_dataset = (train_dataset - clip_min) / (clip_max - clip_min)
    test_dataset = (test_dataset - clip_min) / (clip_max - clip_min)

    save_path = f"{output_path}/healthy_dataset.npy"
    np.save(save_path, train_dataset)
    print(f"Processed training dataset saved as {save_path}. Total of {len(train_dataset)} healthy training images.")

    save_path = f"{output_path}/anomalous_dataset.npy"
    np.save(save_path, test_dataset)
    print(f"Processed testing dataset saved as {save_path}. Total of {len(test_dataset)} anomalous test images.")


def preprocess(server):
    """ Preprocess the BraTS2023 dataset. """

    if server:
        data_path = "/home/polyaxon-data/data1/Lucie"
        output_path = "/home/polyaxon-data/outputs1/lucie_huang/gd_brats23"
    else:
        data_path = "../data/BraTS2023"
        output_path = "../output/GD_BraTS2023"

    create_healthy_and_anomalous_list(data_path, output_path)
    create_healthy_datasets(data_path, output_path)
    create_anomalous_dataset(data_path, output_path, num_test_data=1000)
    # preprocess_datasets(output_path)


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    using_server = False
    preprocess(using_server)
