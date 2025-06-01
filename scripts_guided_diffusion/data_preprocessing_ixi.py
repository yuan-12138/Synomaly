import nibabel as nib
import numpy as np
import glob
import pickle
import cv2

from matplotlib import pyplot as plt


def load_images(data_path, output_path):
    """
    Load images from the LiTS dataset and create healthy and anomalous datasets.

    :param data_path: path to data
    :param output_path: path to save output
    :param num_anomalous_data: number of anomalous data
    :param show_images: boolean to show images
    :return: healthy train and test datasets with liver masks, anomalous test dataset with tumor masks
    """
    volume_files = sorted(glob.glob(f"{data_path}/*.nii.gz"))
    print(f"Number of volumes: {len(volume_files)}")

    healthy_images_raw = generate_datasets(volume_files)

    print(f"Number of healthy images: {len(healthy_images_raw)}")

    if healthy_images_raw.max() < 1:
        clip_min = 0
        clip_max = np.percentile(healthy_images_raw, 99)

        healthy_images = np.clip(healthy_images_raw, clip_min, clip_max)

        healthy_images = (healthy_images - clip_min) / (clip_max - clip_min)
    else:
        healthy_images = healthy_images_raw

    selected_indices = np.random.choice(healthy_images.shape[0], size=500, replace=False)

    healthy_images_test = healthy_images[selected_indices]

    new_size = (240,240)

    resized_healthy_images_test = np.array([cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR) for img in healthy_images_test])

    # save datasets
    # Save healthy training dataset with liver masks
    np.save(f"{output_path}/test_healthy_dataset_ixi.npy", resized_healthy_images_test)
    print(f"Test healthy ixi dataset saved as {output_path}/test_healthy_dataset_ixi.pkl. Number of images: {len(healthy_images_test)}")


def generate_datasets(volume_filenames):
    """
    Generate healthy dataset with liver masks and anomalous dataset with tumor masks.

    :param volume_filenames: abdomen volume filenames
    :param segmentation_filenames: segmentation filenames
    :return: healthy dataset with liver masks and anomalous dataset with tumor masks
    """
    healthy_images = np.empty((0, 256, 256), dtype=np.float64)

    for idx in range(100):
        print(f"Processing volume {idx+1} of {len(volume_filenames)}")

        volume_data = nib.load(volume_filenames[idx]).get_fdata(caching='unchanged')  # unchanged: do not cache .nii file data

        healthy_images = np.concatenate((healthy_images, np.moveaxis(volume_data[:, :, 80:100], 2, 0)), axis=0)

        # free memory
        del volume_data
    return healthy_images

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


if __name__ == "__main__":
    data_path = '/home/camp/Projects/Yuan/Data/IXI_dataset'
    output_path = '/home/camp/Projects/Yuan/thesis_diffusion-main/output/BraTS'
    load_images(data_path, output_path)

    # ixi_data = np.load('/home/camp/Projects/Yuan/thesis_diffusion-main/output/BraTS/test_healthy_dataset.npy')
    #
    # print(ixi_data.max())
    # print(ixi_data.min())
