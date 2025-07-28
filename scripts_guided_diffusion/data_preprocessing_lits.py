import nibabel as nib
import numpy as np
import glob
import pickle

from matplotlib import pyplot as plt


def load_images(data_path, output_path, num_anomalous_data=1000, show_images=False):
    """
    Load images from the LiTS dataset and create healthy and anomalous datasets.

    :param data_path: path to data
    :param output_path: path to save output
    :param num_anomalous_data: number of anomalous data
    :param show_images: boolean to show images
    :return: healthy train and test datasets with liver masks, anomalous test dataset with tumor masks
    """
    volume_files = sorted(glob.glob(f"{data_path}/volumes/*.nii"))
    segmentation_files = sorted(glob.glob(f"{data_path}/segmentations/*.nii"))
    print(f"Number of volumes: {len(volume_files)}")

    healthy_abdomen_images, healthy_liver_masks, anomalous_abdomen_images, tumor_masks = generate_datasets(volume_files, segmentation_files)

    if show_images:
        slice_indices = np.random.randint(0, len(healthy_abdomen_images), 3)

        for idx in slice_indices:
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(healthy_abdomen_images[idx, :, :], cmap='gray')
            plt.title("Healthy abdomen")
            plt.axis('off')
            plt.subplot(2, 2, 2)
            plt.imshow(healthy_liver_masks[idx, :, :], cmap='gray')
            plt.title("Healthy liver mask")
            plt.axis('off')
            plt.subplot(2, 2, 3)
            plt.imshow(anomalous_abdomen_images[idx, :, :], cmap='gray')
            plt.title("Anomalous abdomen")
            plt.axis('off')
            plt.subplot(2, 2, 4)
            plt.imshow(tumor_masks[idx, :, :], cmap='gray')
            plt.title("Tumor mask")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    # Split the healthy dataset into training and validation sets
    shuffle_indices = np.random.permutation(len(healthy_abdomen_images))
    train_test_split_idx = int(len(healthy_abdomen_images) * 0.9)
    train_healthy_abdomen_dataset = healthy_abdomen_images[shuffle_indices[:train_test_split_idx]]
    test_healthy_abdomen_dataset = healthy_abdomen_images[shuffle_indices[train_test_split_idx:]]
    train_healthy_liver_masks = healthy_liver_masks[shuffle_indices[:train_test_split_idx]]
    test_healthy_liver_masks = healthy_liver_masks[shuffle_indices[train_test_split_idx:]]

    # Save random images from the anomalous dataset
    shuffle_indices = np.random.permutation(len(anomalous_abdomen_images))[:num_anomalous_data]
    print(f"Selected {len(shuffle_indices)} random anomalous images from a total of {len(anomalous_abdomen_images)}.")
    test_anomalous_abdomen_dataset = anomalous_abdomen_images[shuffle_indices]
    test_anomalous_tumor_masks = tumor_masks[shuffle_indices]

    # anomalous_idx = np.random.choice(len(anomalous_abdomen_images), num_anomalous_data, replace=False)
    # print(f"Selected {num_anomalous_data} random anomalous images from a total of {len(anomalous_abdomen_images)}.")
    # test_anomalous_dataset = anomalous_abdomen_images[anomalous_idx, :, :]
    # test_anomalous_masks = tumor_masks[anomalous_idx, :, :]

    # idx = 0
    # plt.figure()
    # plt.subplot(2, 3, 1)
    # plt.imshow(train_healthy_abdomen_dataset[idx, :, :], cmap='gray')
    # plt.title("Train healthy abdomen")
    # plt.axis('off')
    # plt.subplot(2, 3, 2)
    # plt.imshow(train_healthy_liver_dataset[idx, :, :], cmap='gray')
    # plt.title("Train healthy liver")
    # plt.axis('off')
    # plt.subplot(2, 3, 3)
    # plt.imshow(train_healthy_liver_masks[idx, :, :], cmap='gray')
    # plt.title("Train healthy liver mask")
    # plt.axis('off')
    # plt.subplot(2, 3, 4)
    # plt.imshow(test_healthy_abdomen_dataset[idx, :, :], cmap='gray')
    # plt.title("Test healthy abdomen")
    # plt.axis('off')
    # plt.subplot(2, 3, 5)
    # plt.imshow(test_healthy_liver_dataset[idx, :, :], cmap='gray')
    # plt.title("Test healthy liver")
    # plt.axis('off')
    # plt.subplot(2, 3, 6)
    # plt.imshow(test_healthy_liver_masks[idx, :, :], cmap='gray')
    # plt.title("Test healthy liver mask")
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    #
    # idx = 0
    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.imshow(test_anomalous_abdomen_dataset[idx, :, :], cmap='gray')
    # plt.title("Test anomalous abdomen")
    # plt.axis('off')
    # plt.subplot(1, 3, 2)
    # plt.imshow(test_anomalous_liver_dataset[idx, :, :], cmap='gray')
    # plt.title("Test anomalous liver")
    # plt.axis('off')
    # plt.subplot(1, 3, 3)
    # plt.imshow(test_anomalous_tumor_masks[idx, :, :], cmap='gray')
    # plt.title("Test tumor mask")
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # save datasets
    # Save healthy training dataset with liver masks
    with open(f"{output_path}/train_healthy_abdomen_dataset.pkl", "wb") as f:
        pickle.dump(train_healthy_abdomen_dataset, f)
    print(f"Train healthy abdomen dataset saved as {output_path}/train_healthy_abdomen_dataset.pkl. Number of images: {len(train_healthy_abdomen_dataset)}")
    with open(f"{output_path}/train_healthy_liver_masks.pkl", "wb") as f:
        pickle.dump(train_healthy_liver_masks, f)
    print(f"Train healthy liver masks saved as {output_path}/train_healthy_liver_masks.pkl. Number of images: {len(train_healthy_liver_masks)}")

    # Save healthy test dataset with liver masks
    with open(f"{output_path}/test_healthy_abdomen_dataset.pkl", "wb") as f:
        pickle.dump(test_healthy_abdomen_dataset, f)
    print(f"Test healthy abdomen dataset saved as {output_path}/test_healthy_abdomen_dataset.pkl. Number of images: {len(test_healthy_abdomen_dataset)}")
    with open(f"{output_path}/test_healthy_liver_masks.pkl", "wb") as f:
        pickle.dump(test_healthy_liver_masks, f)
    print(f"Test healthy liver masks saved as {output_path}/test_healthy_liver_masks.pkl. Number of images: {len(test_healthy_liver_masks)}")

    # Save anomalous test dataset with tumor masks
    with open(f"{output_path}/test_anomalous_abdomen_dataset.pkl", "wb") as f:
        pickle.dump(test_anomalous_abdomen_dataset, f)
    print(f"Anomalous abdomen dataset saved as {output_path}/test_anomalous_abdomen_dataset.pkl. Number of images: {len(test_anomalous_abdomen_dataset)}")
    with open(f"{output_path}/test_anomalous_tumor_masks.pkl", "wb") as f:
        pickle.dump(test_anomalous_tumor_masks, f)
    print(f"Tumor masks saved as {output_path}/test_anomalous_tumor_masks.pkl. Number of images: {len(test_anomalous_tumor_masks)}")

    print(f"Train healthy dataset saved as {output_path}/train_healthy_abdomen_dataset.pkl. Number of images: {len(train_healthy_abdomen_dataset)}")
    print(f"Test healthy dataset saved as {output_path}/test_healthy_abdomen_dataset.pkl. Number of images: {len(test_healthy_abdomen_dataset)}")
    print(f"Anomalous dataset saved as {output_path}/test_anomalous_abdomen_dataset.pkl. Number of images: {len(test_anomalous_abdomen_dataset)}")


def generate_datasets(volume_filenames, segmentation_filenames):
    """
    Generate healthy dataset with liver masks and anomalous dataset with tumor masks.

    :param volume_filenames: abdomen volume filenames
    :param segmentation_filenames: segmentation filenames
    :return: healthy dataset with liver masks and anomalous dataset with tumor masks
    """
    healthy_abdomen_images = np.empty((0, 512, 512), dtype=np.float64)
    healthy_liver_masks = np.empty((0, 512, 512), dtype=np.float64)
    anomalous_abdomen_images = np.empty((0, 512, 512), dtype=np.float64)
    tumor_masks = np.empty((0, 512, 512), dtype=np.float64)

    for idx in range(len(volume_filenames)):
        print(f"Processing volume {idx+1} of {len(volume_filenames)}")

        volume_data = nib.load(volume_filenames[idx]).get_fdata(caching='unchanged')  # unchanged: do not cache .nii file data
        segmentation_data = nib.load(segmentation_filenames[idx]).get_fdata(caching='unchanged')  # unchanged: do not cache .nii file data

        liver_mask = ((segmentation_data == 1) | (segmentation_data == 2))   # liver existing in image
        count_liver = np.sum(segmentation_data == 1, axis=(0, 1))    # healthy liver tissue area
        count_tumor = np.sum(segmentation_data == 2, axis=(0, 1))   # tumor area

        # create healthy, anomalous and tumor masks based on minimal liver and tumor areas
        healthy_mask = (count_liver > 10000) & (count_tumor == 0)
        anomalous_mask = (count_liver + count_tumor >= 10000) & (count_tumor > 1500)
        tumor_mask = (segmentation_data == 2) & anomalous_mask

        volume_data = np.clip(volume_data, 0, 150)  # remove all negative values

        healthy_abdomen_images = np.append(healthy_abdomen_images,
                                           np.moveaxis(volume_data[:, :, healthy_mask], 2, 0).copy(),
                                           axis=0)
        healthy_liver_masks = np.append(healthy_liver_masks,
                                        np.moveaxis(liver_mask[:, :, healthy_mask], 2, 0).copy(),
                                        axis=0)
        anomalous_abdomen_images = np.append(anomalous_abdomen_images,
                                             np.moveaxis(volume_data[:, :, anomalous_mask], 2, 0).copy(),
                                             axis=0)
        tumor_masks = np.append(tumor_masks,
                                np.moveaxis(tumor_mask[:, :, anomalous_mask], 2, 0).copy(),
                                axis=0)

        # free memory
        del volume_data
        del segmentation_data

    return healthy_abdomen_images, healthy_liver_masks, anomalous_abdomen_images, tumor_masks


def preprocess(server):
    """ Preprocess the LiTS dataset. """

    if server:
        data_path = "/home/data/lucie_huang/lits"
        output_path = "/home/data/lucie_huang/lits"
    else:
        data_path = "../data/LiTS"
        output_path = "../output/LiTS"
    load_images(data_path, output_path, num_anomalous_data=1000)


if __name__ == "__main__":
    using_server = True
    preprocess(using_server)
