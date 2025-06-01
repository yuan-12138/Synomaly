import cv2
import nibabel as nib
import numpy as np
import glob
import pickle
import os

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

    all_patients = sorted([f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))])

    volume_files = []
    segmentation_files = []

    for patient in all_patients:
        patient_folder = os.path.join(data_path, patient)
        volume_file = [f for f in os.listdir(patient_folder) if f.endswith("FLAIR.nii.gz")]
        segmentation_file = [f for f in os.listdir(patient_folder) if f.endswith("consensus_gt.nii.gz")]

        volume_files.append(os.path.join(patient_folder, volume_file[0]))
        segmentation_files.append(os.path.join(patient_folder, segmentation_file[0]))
    print(f"Number of volumes: {len(volume_files)}")

    healthy_images_test = generate_datasets(volume_files, segmentation_files)

    # selected_indices = np.random.choice(healthy_images.shape[0], size=500, replace=False)
    #
    # healthy_images_test = healthy_images[selected_indices]

    # new_size = (240, 240)
    #
    # resized_healthy_images_test = np.array(
    #     [cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR) for img in healthy_images_test])

    # save datasets
    # Save healthy training dataset with liver masks
    np.save(f"{output_path}/test_healthy_dataset_3DMRMS.npy", healthy_images_test)
    print(
        f"Test healthy ixi dataset saved as {output_path}/test_healthy_dataset_3DMRMS.pkl. Number of images: {len(healthy_images_test)}")


def generate_datasets(volume_filenames, segmentation_filenames):
    """
    Generate healthy dataset with liver masks and anomalous dataset with tumor masks.

    :param volume_filenames: abdomen volume filenames
    :param segmentation_filenames: segmentation filenames
    :return: healthy dataset with liver masks and anomalous dataset with tumor masks
    """
    healthy_images = np.empty((0, 240, 240), dtype=np.float64)

    for idx in range(len(volume_filenames)):
        print(f"Processing volume {idx+1} of {len(volume_filenames)}")

        nifti_file = nib.load(volume_filenames[idx])

        volume_data = nifti_file.get_fdata(caching='unchanged')  # unchanged: do not cache .nii file data
        segmentation_data = nib.load(segmentation_filenames[idx]).get_fdata(caching='unchanged')  # unchanged: do not cache .nii file data

        brain_slices_range = [310,370]

        healthy_slices = [
            volume_data[:,:,i]
            for i in range(brain_slices_range[0], brain_slices_range[1] + 1)
            if np.all(segmentation_data[:, :, i] == 0)
        ]

        if len(healthy_slices)>0:
            voxel_dimensions = nifti_file.header.get_zooms()
            healthy_slices_resized = np.array(
                [cv2.resize(img, (int(volume_data.shape[1]*voxel_dimensions[1]),int(volume_data.shape[0]*voxel_dimensions[0])), interpolation=cv2.INTER_LINEAR) for img in healthy_slices])

            num, width, height = healthy_slices_resized.shape

            if width > height:
                raise ValueError("Width should not exceed height in the input images.")

            # Calculate padding for width to make it square
            pad_left = (height - width) // 2
            pad_right = height - width - pad_left

            # Pad each image in the sequence
            padded_images = np.pad(
                healthy_slices_resized,
                ((0, 0), (pad_left, pad_right), (0, 0)),  # Pad only along the width dimension
                mode='constant',
                constant_values=0
            )
            print("Patient", idx)

            healthy_images = np.append(healthy_images, padded_images, axis=0)


        #
        # # create healthy, anomalous and tumor masks based on minimal liver and tumor areas
        # healthy_mask = (count_tumor == 0) & (np.arange(len(count_tumor)) >= brain_slices_range[0]) & (np.arange(len(count_tumor)) <= brain_slices_range[1])
        #
        # healthy_slice = np.moveaxis(volume_data[:, :, healthy_mask], 2, 0).copy()
        # if len(healthy_slice) > 0:
        #
        #     healthy_slice = cv2.resize(healthy_slice, (192, 512))
        #
        #     healthy_images = np.append(healthy_images, healthy_slice, axis=0)

        # free memory
        del volume_data
        del segmentation_data

    return healthy_images


if __name__ == "__main__":
    # data_path = '/home/camp/Projects/Yuan/Data/3D_MR_MS'
    # output_path = '/home/camp/Projects/Yuan/thesis_diffusion-main/output/BraTS'
    # load_images(data_path, output_path)

    # nifti_file = nib.load('/home/camp/Projects/Yuan/Data/3D_MR_MS/patient11/patient11_consensus_gt.nii.gz')
    #
    # volume_data = nifti_file.get_fdata(caching='unchanged')
    #
    # header = nifti_file.header
    #
    # # Get the voxel dimensions (scale factor)
    # voxel_dimensions = header.get_zooms()
    #
    # print(header)
    #
    # print(voxel_dimensions)

    #
    # print(f"Volume shape: {volume_data.shape}")
    # print(volume_data.max(), volume_data.min())
    # plt.imshow(volume_data[:,:,320])
    # plt.show()
    #
    # print(np.unique(volume_data))

    ixi_data = np.load('/home/camp/Projects/Yuan/thesis_diffusion-main/output/BraTS/test_healthy_dataset.npy')
    # for img in ixi_data:
    #     plt.imshow(img,cmap='gray')
    #     plt.show()

    print(ixi_data.max())
    print(ixi_data.min())