import pickle
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import InterpolationMode


class RescaleToMinusOneToOne:
    def __call__(self, img):
        min_val = img.min()
        max_val = img.max()
        return 2 * (img - min_val) / (max_val - min_val) - 1

class RescaleToZeroToOne:
    def __call__(self, img):
        min_val = img.min()
        max_val = img.max()
        return (img - min_val) /(max_val - min_val)

# class ColorJitter:
#     """Deprecated because of rescaling to [-1, 1] later"""
#
#     def augment_brightness(self, img, max_percentage=0.1):
#         """
#         Randomly change the brightness of the image
#         :param img: input image
#         :param max_percentage: maximum percentage by which the brightness can be changed, range 0 to 1
#         :return: augmented image
#         """
#         alpha = int((np.random.rand()-0.5) * 2 * 255 * max_percentage)
#         prob = np.random.rand()
#         if prob < 1:
#             brightness_image = img + alpha
#             brightness_image[brightness_image > 255] = 255
#             brightness_image[brightness_image < 0] = 0
#             return brightness_image
#         else:
#             return img
#
#     def augment_contrast(self, img, max_percentage=0.1):
#         alpha = int((np.random.rand()-0.5) * 2 * max_percentage + 1)  # contrast control, 0 < alpha < 1 lower contrast, alpha > 1 higher contrast
#         alpha = 0.2
#         prob = np.random.rand()
#         if prob < 1:
#             contrast_image = img * alpha
#             contrast_image[contrast_image > 255] = 255
#             contrast_image[contrast_image < 0] = 0
#             return contrast_image
#         else:
#             return img
#
#     def __call__(self, img):
#         return self.augment_contrast(self.augment_brightness(img))


class BratsDataset(Dataset):
    def __init__(self, numpy_array, resized_size=128, augment=True):
        self.data = [torch.tensor(numpy_array[i], dtype=torch.float32) for i in range(numpy_array.shape[0])]
        self.resized_size = resized_size

        if augment:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(mode="F"),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),   # Randomly flip image horizontally
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.Resize((self.resized_size, self.resized_size)),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(0.5, 0.5)     # convert to range [-1, 1], (0.5, 0.5, 0.5) for 3D images
                RescaleToMinusOneToOne()
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(mode="F"),
                torchvision.transforms.Resize((self.resized_size, self.resized_size)),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(0.5, 0.5)  # convert to range [-1, 1], (0.5, 0.5, 0.5) for 3D images
                RescaleToMinusOneToOne()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index].unsqueeze(0)
        transformed_sample = self.transform(sample)
        return transformed_sample


class UltrasoundDataset(Dataset):
    def __init__(self, dataset, resized_size=128, augment=True):
        self.data = [torch.from_numpy(arr).float() for arr in dataset]
        self.resized_size = resized_size

        if augment:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(mode="F"),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.Resize((self.resized_size, self.resized_size)),
                torchvision.transforms.ToTensor(),
                RescaleToMinusOneToOne(),
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(mode="F"),
                torchvision.transforms.Resize((self.resized_size, self.resized_size)),
                torchvision.transforms.ToTensor(),
                RescaleToMinusOneToOne()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index].unsqueeze(0)
        transformed_sample = self.transform(sample)

        # plt.figure()
        # plt.subplot(2, 2, 1)
        # plt.imshow(sample.cpu()[0], cmap="gray")
        # plt.title("Original image")
        # plt.subplot(2, 2, 2)
        # plt.imshow(transformed_sample.cpu()[0], cmap="gray")
        # plt.title("Transformed image")
        # plt.subplot(2, 2, 3)
        # plt.hist(sample.cpu().flatten())
        # plt.title("Original histogram")
        # plt.subplot(2, 2, 4)
        # plt.hist(transformed_sample.cpu().flatten())
        # plt.title("Transformed histogram")
        # plt.tight_layout()
        # plt.show()

        return transformed_sample


class UltrasoundDatasetSeg(Dataset):
    def __init__(self, img_set, label_set, resized_size=128, augment=False):
        with open(img_set, "rb") as f:
            anomaly_imgs = pickle.load(f)

        with open(label_set, "rb") as f:
            anomaly_masks = pickle.load(f)

        self.data = [torch.from_numpy(arr/255).float() for arr in anomaly_imgs]
        self.label = [torch.from_numpy(arr).float() for arr in anomaly_masks]
        self.resized_size = resized_size

        if augment:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(mode='F'),  # Convert tensor to PIL image
                # torchvision.transforms.RandomHorizontalFlip(p=0.01),
                # Randomly flip image horizontally with 50% probability
                # torchvision.transforms.RandomVerticalFlip(p=0.5),  # Randomly flip image vertically with 50% probability
                torchvision.transforms.Resize((self.resized_size, self.resized_size)),  # Resize to specified size
                torchvision.transforms.ToTensor(),  # Convert PIL image to tensor
                RescaleToZeroToOne()  # Custom transformation
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(mode="F"),
                torchvision.transforms.Resize((self.resized_size, self.resized_size)),
                torchvision.transforms.ToTensor(),
                RescaleToZeroToOne()
            ])

        self.transform_label = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(mode="F"),
            torchvision.transforms.Resize((self.resized_size, self.resized_size),
                                          interpolation=InterpolationMode.NEAREST),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index].unsqueeze(0)
        transformed_sample = self.transform(sample)

        label = self.label[index].unsqueeze(0)
        transformed_label = self.transform_label(label)

        return transformed_sample, transformed_label

class BratsDatasetSeg(Dataset):
    def __init__(self, img_set, label_set, resized_size=128, augment=False):
        anomaly_imgs = np.load(img_set)
        anomaly_masks = np.load(label_set)

        self.data = [torch.from_numpy(arr/255).float() for arr in anomaly_imgs]
        self.label = [torch.from_numpy(arr).float() for arr in anomaly_masks]
        self.resized_size = resized_size

        if augment:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(mode='F'),  # Convert tensor to PIL image
                # torchvision.transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip image horizontally with 50% probability
                # torchvision.transforms.RandomVerticalFlip(p=0.5),  # Randomly flip image vertically with 50% probability
                torchvision.transforms.Resize((self.resized_size, self.resized_size)),  # Resize to specified size
                torchvision.transforms.ToTensor(),  # Convert PIL image to tensor
                RescaleToZeroToOne()  # Custom transformation
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(mode="F"),
                torchvision.transforms.Resize((self.resized_size, self.resized_size)),
                torchvision.transforms.ToTensor(),
                RescaleToZeroToOne()
            ])

        self.transform_label = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(mode="F"),
            torchvision.transforms.Resize((self.resized_size, self.resized_size), interpolation=InterpolationMode.NEAREST),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index].unsqueeze(0)
        transformed_sample = self.transform(sample)

        label = self.label[index].unsqueeze(0)
        transformed_label = self.transform_label(label)

        return transformed_sample, transformed_label


class LitsDataset(Dataset):
    def __init__(self, dataset, resized_size=64, augment=True):
        self.data = [torch.from_numpy(arr).float() for arr in dataset]
        self.resized_size = resized_size

        if augment:

            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(mode="F"),
                # torchvision.transforms.RandomHorizontalFlip(p=0.5),
                # torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.Resize((self.resized_size, self.resized_size)),
                torchvision.transforms.ToTensor(),
                RescaleToMinusOneToOne(),
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(mode="F"),
                torchvision.transforms.Resize((self.resized_size, self.resized_size)),
                torchvision.transforms.ToTensor(),
                RescaleToMinusOneToOne()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index].unsqueeze(0)
        transformed_sample = self.transform(sample)

        # plt.figure()
        # plt.subplot(2, 2, 1)
        # plt.imshow(sample.cpu()[0], cmap="gray")
        # plt.title("Original image")
        # plt.subplot(2, 2, 2)
        # plt.imshow(transformed_sample.cpu()[0], cmap="gray")
        # plt.title("Transformed image")
        # plt.subplot(2, 2, 3)
        # plt.hist(sample.cpu().flatten())
        # plt.title("Original histogram")
        # plt.subplot(2, 2, 4)
        # plt.hist(transformed_sample.cpu().flatten())
        # plt.title("Transformed histogram")
        # plt.tight_layout()
        # plt.show()

        return transformed_sample


class MaskSet(Dataset):
    def __init__(self, mask_set, resized_size=128):
        self.data = mask_set
        self.resized_size = resized_size

    def transform(self, image):
        resized_image = cv2.resize(image, (self.resized_size, self.resized_size))
        return resized_image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        transformed_sample = self.transform(sample)
        return transformed_sample


def get_training_data(args):
    """
    Get training data for the model.

    :param args: arguments from the config file
    :return: dataloader with image data and synth_anomaly_mask_loader with regions where synthetic anomalies are added
    """

    if args["dataset"].lower() == "brats23":
        data = np.load(f"{args['output_path']}/train_healthy_dataset.npy")

        data = data[:args['num_training_data']]
        print(f"Training on {len(data)} images")

        dataset = BratsDataset(data, args['img_size'], augment=True)
        dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)
        synth_anomaly_mask_loader = None


    elif args["dataset"].lower() == "ultrasound":

        with open(f"{args['output_path']}/train_healthy_dataset.pkl", "rb") as f:

            data = pickle.load(f)

        data = data[:args['num_training_data']]

        print(f"Training on {len(data)} images")

        dataset = UltrasoundDataset(data, args['img_size'], augment=True)

        dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)

        if args["apply_mask"]:

            # vessel mask

            center_y, center_x = args['img_size'] // 2, args['img_size'] // 2

            Y, X = np.ogrid[:args['img_size'], :args['img_size']]

            dist_from_center = (X - center_x) ** 2 + (Y - center_y) ** 2

            radius = args['img_size'] // 2.2

            single_vessel_mask = (dist_from_center <= radius ** 2).astype(int)

            vessel_masks = [single_vessel_mask for _ in range(len(data))]

            vessel_mask_set = UltrasoundDataset(vessel_masks, args['img_size'], augment=True)

            synth_anomaly_mask_loader = DataLoader(vessel_mask_set, batch_size=args['batch_size'], shuffle=True)
        else:

            synth_anomaly_mask_loader = None


    elif args["dataset"].lower() == "lits":

        with open(f"{args['output_path']}/train_healthy_abdomen_dataset.pkl", "rb") as f:

            data = pickle.load(f)

        data = data[:args['num_training_data']]

        print(f"Training on {len(data)} images")

        dataset = LitsDataset(data, args['img_size'], augment=True)

        dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True,
                                generator=torch.Generator().manual_seed(314))

        if args['apply_mask']:

            with open(f"{args['output_path']}/train_healthy_liver_masks.pkl", "rb") as f:

                liver_masks = pickle.load(f)

            liver_masks = liver_masks[:args['num_training_data']]

            liver_mask_set = LitsDataset(liver_masks, args['img_size'], augment=True)

            synth_anomaly_mask_loader = DataLoader(liver_mask_set, batch_size=args['batch_size'], shuffle=True,
                                                   generator=torch.Generator().manual_seed(314))

        else:

            synth_anomaly_mask_loader = None

    else:

        raise ValueError("Invalid dataset")

    return dataloader, synth_anomaly_mask_loader


def get_test_data(args):
    """
    Get test data.

    :param args: arguments from the config file
    """

    data, anomaly_masks = None, None
    if args["dataset"].lower() == "brats23":
        if args["Version"].lower() == "new":
            print("Test on new dataset.")
            if args['mode'] == "anomalous":
                data = np.load(f"{args['output_path']}/test_anomalous_dataset_new.npy")
                anomaly_masks = np.load(f"{args['output_path']}/test_anomalous_masks_new.npy")
            elif args['mode'] == "healthy":
                data = np.load(f"{args['output_path']}/test_healthy_dataset_new.npy")
                anomaly_masks = np.zeros_like(data, dtype=np.float64)
            elif args['mode'] == "both":
                data_anomalous = np.load(f"{args['output_path']}/test_anomalous_dataset_new.npy")
                anomaly_masks_anomalous = np.load(f"{args['output_path']}/test_anomalous_masks_new.npy")
                data_healthy = np.load(f"{args['output_path']}/test_healthy_dataset_new.npy")
                anomaly_masks_healthy = np.zeros_like(data_healthy, dtype=np.float64)
                data = np.concatenate((data_anomalous, data_healthy), axis=0)
                anomaly_masks = np.concatenate((anomaly_masks_anomalous, anomaly_masks_healthy), axis=0)
        else:
            if args['mode'] == "anomalous":
                data = np.load(f"{args['output_path']}/test_anomalous_dataset.npy")
                anomaly_masks = np.load(f"{args['output_path']}/test_anomalous_masks.npy")
            elif args['mode'] == "healthy":
                data = np.load(f"{args['output_path']}/test_healthy_dataset_ixi.npy")
                anomaly_masks = np.zeros_like(data, dtype=np.float64)
            elif args['mode'] == "both":
                data_anomalous = np.load(f"{args['output_path']}/test_anomalous_dataset.npy")
                anomaly_masks_anomalous = np.load(f"{args['output_path']}/test_anomalous_masks.npy")
                data_healthy = np.load(f"{args['output_path']}/test_healthy_dataset.npy")
                anomaly_masks_healthy = np.zeros_like(data_healthy, dtype=np.float64)
                data = np.concatenate((data_anomalous, data_healthy), axis=0)
                anomaly_masks = np.concatenate((anomaly_masks_anomalous, anomaly_masks_healthy), axis=0)

        # random.seed(314)  # 314, 222
        # sample_indices = random.sample(range(len(data)), min(args['num_test_img'], len(data)))
        # data = data[sample_indices]
        # anomaly_masks = anomaly_masks[sample_indices]
        print(f"Testing on {len(data)} randomly selected images. Mode: {args['mode']}")

        dataset = BratsDataset(data, args['img_size'], augment=False)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        anomaly_masks = MaskSet(anomaly_masks, args['img_size'])

    elif args["dataset"].lower() == "ultrasound":
        if args["Version"].lower() == "new":
            print("Test on new dataset.")
            if args['mode'] == "anomalous":
                with open(f"{args['output_path']}/test_anomalous_dataset.pkl", "rb") as f:
                    data = pickle.load(f)
                with open(f"{args['output_path']}/test_anomalous_masks.pkl", "rb") as f:
                    anomaly_masks = pickle.load(f)
            elif args['mode'] == "healthy":
                with open(f"{args['output_path']}/test_healthy_dataset_new.pkl", "rb") as f:
                    data = pickle.load(f)
                anomaly_masks = [np.zeros_like(data[i]) for i in range(len(data))]
            elif args['mode'] == "both":
                with open(f"{args['output_path']}/test_anomalous_dataset.pkl", "rb") as f:
                    data_anomalous = pickle.load(f)
                with open(f"{args['output_path']}/test_anomalous_masks.pkl", "rb") as f:
                    anomaly_masks_anomalous = pickle.load(f)
                with open(f"{args['output_path']}/test_healthy_dataset_new.pkl", "rb") as f:
                    data_healthy = pickle.load(f)
                anomaly_masks_healthy = [np.zeros_like(data_healthy[i]) for i in range(len(data_healthy))]
                data = data_anomalous+data_healthy
                anomaly_masks = anomaly_masks_anomalous+anomaly_masks_healthy
        else:
            if args['mode'] == "anomalous":
                with open(f"{args['output_path']}/test_anomalous_dataset.pkl", "rb") as f:
                    data = pickle.load(f)
                with open(f"{args['output_path']}/test_anomalous_masks.pkl", "rb") as f:
                    anomaly_masks = pickle.load(f)
            elif args['mode'] == "healthy":
                with open(f"{args['output_path']}/test_healthy_dataset.pkl", "rb") as f:
                    data = pickle.load(f)
                anomaly_masks = [np.zeros_like(data[i]) for i in range(len(data))]
            elif args['mode'] == "both":
                with open(f"{args['output_path']}/test_anomalous_dataset.pkl", "rb") as f:
                    data_anomalous = pickle.load(f)
                with open(f"{args['output_path']}/test_anomalous_masks.pkl", "rb") as f:
                    anomaly_masks_anomalous = pickle.load(f)
                with open(f"{args['output_path']}/test_healthy_dataset.pkl", "rb") as f:
                    data_healthy = pickle.load(f)
                anomaly_masks_healthy = [np.zeros_like(data_healthy[i]) for i in range(len(data_healthy))]
                data = data_anomalous+data_healthy
                anomaly_masks = anomaly_masks_anomalous+anomaly_masks_healthy

        random.seed(314)  # 314, 222
        sample_indices = random.sample(range(len(data)), min(args['num_test_img'], len(data)))
        data = [data[i] for i in sample_indices]
        anomaly_masks = [anomaly_masks[i] for i in sample_indices]
        print(f"Testing on {len(data)} randomly selected images. Mode: {args['mode']}")

        dataset = UltrasoundDataset(data, args['img_size'], augment=False)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        anomaly_masks = MaskSet(anomaly_masks, args['img_size'])

    # elif args["dataset"].lower() == "ultrasound":
    #     if args['mode'] == "anomalous":
    #         with open(f"{args['output_path']}/anomalous_dataset_unet_cropped.pkl", "rb") as f:
    #             data = pickle.load(f)
    #         with open(f"{args['output_path']}/anomalous_masks_unet_cropped.pkl", "rb") as f:
    #             anomaly_masks = pickle.load(f)
    #     elif args['mode'] == "healthy":
    #         with open(f"{args['output_path']}/test_healthy_dataset.pkl", "rb") as f:
    #             data = pickle.load(f)
    #         anomaly_masks = [np.zeros_like(data[i]) for i in range(len(data))]
    #     elif args['mode'] == "both":
    #         with open(f"{args['output_path']}/test_anomalous_dataset.pkl", "rb") as f:
    #             data_anomalous = pickle.load(f)
    #         with open(f"{args['output_path']}/test_anomalous_masks.pkl", "rb") as f:
    #             anomaly_masks_anomalous = pickle.load(f)
    #         with open(f"{args['output_path']}/test_healthy_dataset.pkl", "rb") as f:
    #             data_healthy = pickle.load(f)
    #         anomaly_masks_healthy = [np.zeros_like(data_healthy[i]) for i in range(len(data_healthy))]
    #         data = data_anomalous + data_healthy
    #         anomaly_masks = anomaly_masks_anomalous + anomaly_masks_healthy
    #
    #     random.seed(314)  # 314, 222
    #     sample_indices = random.sample(range(len(data)), min(args['num_test_img'], len(data)))
    #     data = [data[i] for i in sample_indices]
    #     anomaly_masks = [anomaly_masks[i] for i in sample_indices]
    #     print(f"Testing on {len(data)} randomly selected images. Mode: {args['mode']}")
    #
    #     dataset = UltrasoundDataset(data, args['img_size'], augment=False)
    #     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    #
    #     anomaly_masks = MaskSet(anomaly_masks, args['img_size'])

    elif args["dataset"].lower() == "lits":
        if args["Version"].lower() == "new":
            print("Test on new dataset.")
            if args['mode'] == "anomalous":
                with open(f"{args['output_path']}/test_anomalous_abdomen_dataset_new.pkl", "rb") as f:
                    data = pickle.load(f)
                with open(f"{args['output_path']}/test_anomalous_tumor_masks_new.pkl", "rb") as f:
                    anomaly_masks = pickle.load(f)
            elif args['mode'] == "healthy":
                with open(f"{args['output_path']}/test_healthy_abdomen_dataset_new.pkl", "rb") as f:
                    data = pickle.load(f)
                anomaly_masks = [np.zeros_like(data[i]) for i in range(len(data))]
            elif args['mode'] == "both":
                with open(f"{args['output_path']}/test_anomalous_abdomen_dataset_new.pkl", "rb") as f:
                    data_anomalous = pickle.load(f)
                with open(f"{args['output_path']}/test_anomalous_tumor_masks_new.pkl", "rb") as f:
                    anomaly_masks_anomalous = pickle.load(f)
                with open(f"{args['output_path']}/test_healthy_abdomen_dataset_new.pkl", "rb") as f:
                    data_healthy = pickle.load(f)
                anomaly_masks_healthy = [np.zeros_like(data_healthy[i]) for i in range(len(data_healthy))]
                data = np.concatenate((data_anomalous, data_healthy), axis=0)
                anomaly_masks = np.concatenate((anomaly_masks_anomalous, anomaly_masks_healthy), axis=0)
        else:
            if args['mode'] == "anomalous":
                with open(f"{args['output_path']}/test_anomalous_abdomen_dataset.pkl", "rb") as f:
                    data = pickle.load(f)
                with open(f"{args['output_path']}/test_anomalous_tumor_masks.pkl", "rb") as f:
                    anomaly_masks = pickle.load(f)
            elif args['mode'] == "healthy":
                with open(f"{args['output_path']}/test_healthy_abdomen_dataset.pkl", "rb") as f:
                    data = pickle.load(f)
                anomaly_masks = [np.zeros_like(data[i]) for i in range(len(data))]
            elif args['mode'] == "both":
                with open(f"{args['output_path']}/test_anomalous_abdomen_dataset.pkl", "rb") as f:
                    data_anomalous = pickle.load(f)
                with open(f"{args['output_path']}/test_anomalous_tumor_masks.pkl", "rb") as f:
                    anomaly_masks_anomalous = pickle.load(f)
                with open(f"{args['output_path']}/test_healthy_abdomen_dataset.pkl", "rb") as f:
                    data_healthy = pickle.load(f)
                anomaly_masks_healthy = [np.zeros_like(data_healthy[i]) for i in range(len(data_healthy))]
                data = np.concatenate((data_anomalous, data_healthy), axis=0)
                anomaly_masks = np.concatenate((anomaly_masks_anomalous, anomaly_masks_healthy), axis=0)

        random.seed(314)  # 222
        sample_indices = random.sample(range(len(data)), min(args['num_test_img'], len(data)))
        data = [data[i] for i in sample_indices]
        anomaly_masks = [anomaly_masks[i] for i in sample_indices]
        print(f"Testing on {len(data)} randomly selected images. Mode: {args['mode']}")

        dataset = UltrasoundDataset(data, args['img_size'], augment=False)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        anomaly_masks = MaskSet(anomaly_masks, args['img_size'])

    else:
        raise ValueError("Invalid dataset")

    return dataloader, anomaly_masks

def get_test_data_(dataset_name,mode,num_test_img):
    """
    Get test data.

    :param args: arguments from the config file
    """

    data, anomaly_masks = None, None
    if dataset_name.lower() == "brats23":
        if mode == "anomalous":
            data = np.load(f"../output/BraTS/test_anomalous_dataset.npy")
            anomaly_masks = np.load(f"../output/BraTS/test_anomalous_masks.npy")
        elif mode == "healthy":
            data = np.load(f"../output/BraTS/test_healthy_dataset.npy")
            anomaly_masks = np.zeros_like(data, dtype=np.float64)
        elif mode == "both":
            data_anomalous = np.load(f"../output/BraTS/test_anomalous_dataset.npy")
            anomaly_masks_anomalous = np.load(f"../output/BraTS/test_anomalous_masks.npy")
            data_healthy = np.load(f"../output/BraTS/test_healthy_dataset.npy")
            anomaly_masks_healthy = np.zeros_like(data_healthy, dtype=np.float64)
            data = np.concatenate((data_anomalous, data_healthy), axis=0)
            anomaly_masks = np.concatenate((anomaly_masks_anomalous, anomaly_masks_healthy), axis=0)

        random.seed(314)  # 314, 222
        sample_indices = random.sample(range(len(data)), min(num_test_img, len(data)))
        data = data[sample_indices]
        anomaly_masks = anomaly_masks[sample_indices]
        print(f"Testing on {len(data)} randomly selected images. Mode: {mode}")

        dataset = BratsDataset(data, 128, augment=False)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        anomaly_masks = MaskSet(anomaly_masks, 128)

    elif dataset_name.lower() == "ultrasound":
        if mode == "anomalous":
            with open(f"../output/Ultrasound/test_anomalous_dataset.pkl", "rb") as f:
                data = pickle.load(f)
            with open(f"../output/Ultrasound/test_anomalous_masks.pkl", "rb") as f:
                anomaly_masks = pickle.load(f)
        elif mode == "healthy":
            with open(f"../output/Ultrasound/test_healthy_dataset.pkl", "rb") as f:
                data = pickle.load(f)
            anomaly_masks = [np.zeros_like(data[i]) for i in range(len(data))]
        elif mode == "both":
            with open(f"../output/Ultrasound/test_anomalous_dataset.pkl", "rb") as f:
                data_anomalous = pickle.load(f)
            with open(f"../Ultrasound/test_anomalous_masks.pkl", "rb") as f:
                anomaly_masks_anomalous = pickle.load(f)
            with open(f"../output/Ultrasound/test_healthy_dataset.pkl", "rb") as f:
                data_healthy = pickle.load(f)
            anomaly_masks_healthy = [np.zeros_like(data_healthy[i]) for i in range(len(data_healthy))]
            data = np.concatenate((data_anomalous, data_healthy), axis=0)
            anomaly_masks = np.concatenate((anomaly_masks_anomalous, anomaly_masks_healthy), axis=0)

        random.seed(314)  # 314, 222
        sample_indices = random.sample(range(len(data)), min(num_test_img, len(data)))
        data = [data[i] for i in sample_indices]
        anomaly_masks = [anomaly_masks[i] for i in sample_indices]
        print(f"Testing on {len(data)} randomly selected images. Mode: {mode}")

        dataset = UltrasoundDataset(data, 128, augment=False)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        anomaly_masks = MaskSet(anomaly_masks, 128)

    elif dataset_name.lower() == "lits":
        if mode == "anomalous":
            with open(f"../output/LiTS/test_anomalous_abdomen_dataset.pkl", "rb") as f:
                data = pickle.load(f)
            with open(f"../output/LiTS/test_anomalous_tumor_masks.pkl", "rb") as f:
                anomaly_masks = pickle.load(f)
        elif mode == "healthy":
            with open(f"../output/LiTS/test_healthy_abdomen_dataset.pkl", "rb") as f:
                data = pickle.load(f)
            anomaly_masks = [np.zeros_like(data[i]) for i in range(len(data))]
        elif mode == "both":
            with open(f"../output/LiTS/test_anomalous_abdomen_dataset.pkl", "rb") as f:
                data_anomalous = pickle.load(f)
            with open(f"../output/LiTS/test_anomalous_tumor_masks.pkl", "rb") as f:
                anomaly_masks_anomalous = pickle.load(f)
            with open(f"../output/LiTS/test_healthy_abdomen_dataset.pkl", "rb") as f:
                data_healthy = pickle.load(f)
            anomaly_masks_healthy = [np.zeros_like(data_healthy[i]) for i in range(len(data_healthy))]
            data = np.concatenate((data_anomalous, data_healthy), axis=0)
            anomaly_masks = np.concatenate((anomaly_masks_anomalous, anomaly_masks_healthy), axis=0)

        random.seed(314)  # 222
        sample_indices = random.sample(range(len(data)), min(num_test_img, len(data)))
        data = [data[i] for i in sample_indices]
        anomaly_masks = [anomaly_masks[i] for i in sample_indices]
        print(f"Testing on {len(data)} randomly selected images. Mode: {mode}")

        dataset = UltrasoundDataset(data, 128, augment=False)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        anomaly_masks = MaskSet(anomaly_masks, 128)

    else:
        raise ValueError("Invalid dataset")

    return dataloader, anomaly_masks
