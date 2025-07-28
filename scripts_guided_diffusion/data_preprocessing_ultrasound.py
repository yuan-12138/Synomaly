import json

import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import torch
import pickle
from itertools import combinations

def crop_to_vessel(img, vessel_mask, crop_factor=1.0, width_vario=0.5, height_vario=0.5):
    """
    Crop the image to the vessel mask.

    :param img: input image
    :param vessel_mask: vessel mask
    :param crop_factor: crop factor, 1.0 for exact bounding box of vessel mask, >1.0 for larger areas
    :param width_vario: positional shift of vessel in horizontal direction
    :param height_vario: positional shift of vessel in vertical direction
    :return: cropped image
    """

    # Find contours in the vessel_mask
    contours, _ = cv2.findContours(vessel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box around the vessel_mask
    x, y, w, h = cv2.boundingRect(contours[0])

    # Calculate the center of the bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    # Determine the size of the square
    cropped_width = w * crop_factor
    cropped_height = h * crop_factor

    x1 = max(0, center_x - int(cropped_width * width_vario))
    y1 = max(0, center_y - int(cropped_height * height_vario))
    x2 = min(img.shape[1], center_x + int(cropped_width * (1-width_vario)))
    y2 = min(img.shape[0], center_y + int(cropped_height * (1-height_vario)))

    # Crop the image to the square
    return img[y1:y2, x1:x2]


def read_selection_from_json(file_path):
    """
    Reads selection criteria from a JSON file.

    :param file_path: path to JSON file
    :return: selection criteria
    """
    with open(file_path, 'r') as f:
        selection_data = json.load(f)

    selection = {}
    for subfolder, data in selection_data.items():
        ranges = [(int(start), int(end)) for range_str in data["ranges"] for start, end in [range_str.split('-')]]
        step_size = data["step_size"]
        selection[subfolder] = {"ranges": ranges, "step_size": step_size}

    return selection


def create_healthy_datasets(healthy_selection_json, dataset_path, show_images=False):
    """
    Create healthy datasets from the selected healthy images.

    :param healthy_selection_json: selected healthy images
    :param dataset_path: path to data
    :param show_images: boolean to show images
    :return: healthy training and test datasets
    """
    healthy_selection = read_selection_from_json(healthy_selection_json)

    image_dataset = []

    number_of_images = []
    folders = []

    for data_path, selection in healthy_selection.items():
        print(f"Processing {data_path}")
        current_image_dataset = []
        slice_ranges = selection["ranges"]
        image_step_size = selection["step_size"]

        for start_index, end_index in slice_ranges:
            image_files = sorted(glob.glob(f"/home/camp/Projects/Yuan/Data/Ultrasound_synomaly/healthy/{data_path}/img/*.png"))[start_index - 1: end_index]
            vessel_files = sorted(glob.glob(f"/home/camp/Projects/Yuan/Data/Ultrasound_synomaly/healthy/{data_path}/vessel/*.png"))[start_index - 1: end_index]

            for i, img_id in enumerate(range(0, len(image_files), image_step_size)):
                img = cv2.imread(image_files[img_id])[:, :, 0]
                vessel_mask = cv2.imread(vessel_files[img_id], cv2.IMREAD_GRAYSCALE)

                # augmentation
                crop_factor = np.random.uniform(1.1, 1.5)
                width_vario = np.random.uniform(0.46, 0.54)
                height_vario = np.random.uniform(0.46, 0.54)

                cropped_image = crop_to_vessel(img, vessel_mask, crop_factor, width_vario, height_vario)
                current_image_dataset.append(cropped_image)
                # if show_images:
                #     plt.figure()
                #     plt.subplot(1, 3, 1)
                #     plt.imshow(img, cmap="gray")
                #     plt.title("Original Image")
                #     plt.subplot(1, 3, 2)
                #     plt.imshow(vessel_mask, cmap="gray")
                #     plt.title("Vessel mask")
                #     plt.subplot(1, 3, 3)
                #     plt.imshow(cropped_image, cmap="gray")
                #     plt.title("Cropped Image")
                #     plt.tight_layout()
                #     plt.suptitle(f"{data_path}, {img_id}, {i}")
                #     plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
                #     plt.show()


        image_dataset = image_dataset+current_image_dataset

        number_of_images.append(len(current_image_dataset))
        folders.append(data_path)

    subset_indices, remaining_indices = divide_list_by_index(number_of_images, 7306)

    subset_numbers = [number_of_images[i] for i in subset_indices]
    remaining_numbers = [number_of_images[i] for i in remaining_indices]
    print("Subset close to target (values):", np.sum(subset_numbers))
    print("Remaining numbers (values):", np.sum(remaining_numbers))
    print(number_of_images)



    # # Split the dataset into training and validation sets
    # np.random.shuffle(image_dataset)
    # split_idx = int(len(image_dataset) * 0.9)
    # train_healthy_dataset = image_dataset[:split_idx]
    # test_healthy_dataset = image_dataset[split_idx:]
    #
    # with open(f"{dataset_path}/train_healthy_dataset.pkl", "wb") as f:
    #     pickle.dump(train_healthy_dataset, f)
    # with open(f"{dataset_path}/test_healthy_dataset.pkl", "wb") as f:
    #     pickle.dump(test_healthy_dataset, f)
    #
    # print(f"Train healthy dataset saved as {dataset_path}/train_healthy_dataset.pkl. Number of images: {len(train_healthy_dataset)}")
    # print(f"Test healthy dataset saved as {dataset_path}/test_healthy_dataset.pkl. Number of images: {len(test_healthy_dataset)}")

def create_anomalous_datasets(selected_anomalous_paths, dataset_path, show_images=False):
    """
    Create anomalous dataset.

    :param selected_anomalous_paths: paths to selected anomalous images
    :param dataset_path: path to data
    :param show_images: boolean to show images
    :return: anomalous test dataset with plaque masks
    """
    image_dataset = []
    plaque_dataset = []
    for data_path in selected_anomalous_paths:
        image_files = sorted(glob.glob(f"{data_path}/img/*.png"))
        vessel_files = sorted(glob.glob(f"{data_path}/vessel/*.png"))
        plaque_files = sorted(glob.glob(f"{data_path}/plaque/*.png"))

        for img, vessel, plaque in zip(image_files, vessel_files, plaque_files):
            plaque_mask = cv2.imread(plaque, cv2.IMREAD_GRAYSCALE)
            if np.sum(plaque_mask) < 2000:
                continue

            img = cv2.imread(img)[:, :, 0]
            vessel_mask = cv2.imread(vessel, cv2.IMREAD_GRAYSCALE)

            # slight random crop augmentations
            crop_factor = np.random.uniform(1.1, 1.5)
            width_vario = np.random.uniform(0.46, 0.54)
            height_vario = np.random.uniform(0.46, 0.54)

            # cropped_image = crop_to_vessel(img, vessel_mask, crop_factor, width_vario, height_vario)
            # cropped_plaque_mask = crop_to_vessel(plaque_mask, vessel_mask, crop_factor, width_vario, height_vario)

            if show_images:
                plt.figure(figsize=(20, 7))
                plt.subplot(1, 4, 1)
                plt.imshow(img, cmap="gray")
                plt.title("Original Image")
                plt.subplot(1, 4, 2)
                plt.imshow(vessel_mask, cmap="gray")
                plt.title("Vessel Mask")
                plt.subplot(1, 4, 3)
                plt.imshow(cropped_image, cmap="gray")
                plt.title("Cropped Image")
                plt.subplot(1, 4, 4)
                plt.imshow(cropped_plaque_mask, cmap="gray")
                plt.title("Plaque")
                plt.tight_layout()
                plt.suptitle(f"{data_path}")
                plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
                plt.show()

            image_dataset.append(img)
            plaque_dataset.append(plaque_mask)

    with open(f"{dataset_path}/anomalous_dataset_unet.pkl", "wb") as f:
        pickle.dump(image_dataset, f)
    with open(f"{dataset_path}/anomalous_masks_unet.pkl", "wb") as f:
        pickle.dump(plaque_dataset, f)

    print(f"Anomalous dataset saved as {dataset_path}/anomalous_dataset_unet.pkl. Number of images: {len(image_dataset)}")
    print(f"Plaque dataset saved as {dataset_path}/anomalous_masks_unet.pkl. Number of images: {len(plaque_dataset)}")


def preprocess(server):
    """ Preprocess the ultrasound carotid dataset. """

    if server:
        output_path = "/home/polyaxon-data/outputs1/lucie_huang/Ultrasound"
    else:
        output_path = "/home/camp/Projects/Yuan/thesis_diffusion-main/output/Ultrasound"

    # 397 healthy images for [50....50, 10....10]
    # selected_healthy_data = [
    #     ("../data/Ultrasound/healthy/continuous/carotid/1", (1, 2525), 50),
    #     ("../data/Ultrasound/healthy/continuous/carotid/2", (1, 1300), 50),
    #     ("../data/Ultrasound/healthy/continuous/carotid/3", (1, 550), 50),
    #     ("../data/Ultrasound/healthy/continuous/carotid/4", (100, 3242), 50),
    #     ("../data/Ultrasound/healthy/continuous/carotid/5", (1, 1710), 50),
    #     ("../data/Ultrasound/healthy/continuous/carotid/6", (1, 836), 50),
    #     ("../data/Ultrasound/healthy/continuous/carotid/7", (100, 2196), 50),
    #     ("../data/Ultrasound/healthy/continuous/carotid/8", (1, 1082), 50),
    #     ("../data/Ultrasound/healthy/continuous/carotid/9", (1, 510), 50),
    #     ("../data/Ultrasound/healthy/lucie/1", (50, 459), 10),
    #     ("../data/Ultrasound/healthy/lucie/2", (1, 420), 10),
    #     ("../data/Ultrasound/healthy/lucie/3", (100, 320), 10),
    #     ("../data/Ultrasound/healthy/lucie/3", (380, 530), 10),
    #     ("../data/Ultrasound/healthy/lucie/3", (760, 790), 10),
    # ]

    # load json file
    # 5240 healthy images
    # 8118 healthy images
    healthy_selection_json = f"{output_path}/healthy_image_selection.json"

    # 545 anomalous images
    # selected_anomalous_data = [
    #     "../data/Ultrasound/unhealthy/Moustafa/16.11.2022/patient_1",
    #     # "../data/Ultrasound/unhealthy/Moustafa/16.12.2022/patient_2",
    #     # "../data/Ultrasound/unhealthy/Moustafa/16.12.2022/patient_4",
    #     "../data/Ultrasound/unhealthy/Moustafa/18.10.2022/1",
    #     "../data/Ultrasound/unhealthy/Moustafa/18.10.2022/2",
    #     "../data/Ultrasound/unhealthy/Moustafa/18.10.2022/3",
    #     "../data/Ultrasound/unhealthy/Moustafa/19.01.2023/2",
    #     "../data/Ultrasound/unhealthy/Moustafa/19.01.2023/4",
    #     "../data/Ultrasound/unhealthy/Moustafa/19.01.2023/5",
    #     "../data/Ultrasound/unhealthy/Moustafa/19.01.2023/7",
    # ]

    selected_anomalous_data = [
        "/home/camp/Projects/Yuan/Data/carotid_plaques/Moustafa_Annotation/16.11.2022/patient_1",
        "/home/camp/Projects/Yuan/Data/carotid_plaques/Moustafa_Annotation/18.10.2022/1",
        "/home/camp/Projects/Yuan/Data/carotid_plaques/Moustafa_Annotation/18.10.2022/2",
        "/home/camp/Projects/Yuan/Data/carotid_plaques/Moustafa_Annotation/18.10.2022/3",
        "/home/camp/Projects/Yuan/Data/carotid_plaques/Moustafa_Annotation/19.01.2023/2",
        "/home/camp/Projects/Yuan/Data/carotid_plaques/Moustafa_Annotation/19.01.2023/4",
        "/home/camp/Projects/Yuan/Data/carotid_plaques/Moustafa_Annotation/19.01.2023/5",
        "/home/camp/Projects/Yuan/Data/carotid_plaques/Moustafa_Annotation/19.01.2023/7",
    ]

    # create_healthy_datasets(healthy_selection_json, output_path, show_images=False)
    create_anomalous_datasets(selected_anomalous_data, output_path, show_images=False)


if __name__ == "__main__":
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # using_server = False
    # preprocess(using_server)

    # jason_file_path = '/home/camp/Projects/Yuan/thesis_diffusion-main/output/Ultrasound/healthy_image_selection.json'
    # selection = read_selection_from_json(jason_file_path)
    #
    # print(selection)

    healthy_selection_json = '/home/camp/Projects/Yuan/thesis_diffusion-main/output/Ultrasound/healthy_image_selection.json'
    dataset_path = '1'
    create_healthy_datasets(healthy_selection_json, dataset_path, show_images=False)