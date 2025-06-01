from unet_seg import UNet
from data_loader import UltrasoundDatasetSeg, BratsDatasetSeg
import torch
import matplotlib.pyplot as plt

import numpy as np
import os
import shutil
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split, Subset

lr = 5e-4
b1 = 0.5
b2 = 0.999
min_lr = 1e-5
num_epoch = 4000
batch_size = 16
start_epoch = 0
# save_dir = "../output/Ultrasound/seg_model"
# img_file = '../output/Ultrasound/test_anomalous_dataset.pkl'
# label_file = '../output/Ultrasound/test_anomalous_masks.pkl'
#
# train_dataset = UltrasoundDatasetSeg(img_file, label_file,128,False)
# test_dataset = UltrasoundDatasetSeg(img_file, label_file,128,False)

# save_dir = "../output/BraTS/seg_model"
# img_file = '../output/BraTS/test_anomalous_dataset.npy'
# label_file = '../output/BraTS/test_anomalous_masks.npy'
#
# train_dataset = BratsDatasetSeg(img_file, label_file,128,False)
# test_dataset = BratsDatasetSeg(img_file, label_file,128,False)

save_dir = "../output/LiTS/seg_model"
img_file = '../output/LiTS/test_anomalous_abdomen_dataset.pkl'
label_file = '../output/LiTS/test_anomalous_tumor_masks.pkl'

train_dataset = UltrasoundDatasetSeg(img_file, label_file,128,False)
test_dataset = UltrasoundDatasetSeg(img_file, label_file,128,False)

num_training_data = int(len(train_dataset) * 0.5)

train_dataset = Subset(train_dataset, list(range(0, num_training_data)))
test_dataset = Subset(test_dataset, list(range(num_training_data, len(test_dataset))))

# Create DataLoader for training and test sets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

dataset_name = 'brats'
# pretrained_path = '../saved_models/brats23_checkpoint.pth'
# pretrained_path = '../output/BraTS/seg_model/brats_model_best.pth'
pretrained_path = None

model = UNet(init_features=64).to('cuda')

optimizer = torch.optim.Adam(model.parameters(),lr=lr, betas=(b1, b2))

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

best_loss = np.inf

if pretrained_path is not None:
    checkpoint = torch.load(pretrained_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]

def save_checkpoint(state, is_best, outdir, model_name):

	if not os.path.exists(outdir):
		os.makedirs(outdir)

	checkpoint_file = os.path.join(outdir, model_name+'_checkpoint.pth')
	best_file = os.path.join(outdir, model_name+'_model_best.pth')
	torch.save(state, checkpoint_file)
	if is_best:
		shutil.copyfile(checkpoint_file, best_file)

def dice_loss(pred, target, smooth=1e-6):
    """
    Calculate Dice Loss.

    Parameters:
    - pred (torch.Tensor): Predicted tensor, with values between 0 and 1. Shape: [B, C, H, W]
    - target (torch.Tensor): Ground truth tensor, with values 0 or 1. Shape: [B, C, H, W]
    - smooth (float): Smoothing constant to avoid division by zero.

    Returns:
    - loss (torch.Tensor): Calculated Dice Loss.
    """
    # Apply sigmoid if needed to get predictions in range [0, 1]
    pred = torch.sigmoid(pred) if pred.max() > 1.0 else pred
    target = (target >= 1).int()

    # Flatten the tensors for calculating the Dice coefficient
    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)

    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    # Calculate the Dice coefficient
    dice_coeff = (2. * intersection + smooth) / (union + smooth)

    # Dice loss is 1 - Dice coefficient
    dice_loss = 1 - dice_coeff

    # Return mean Dice loss for the batch
    return dice_loss.mean()


# train_dataset = UltrasoundDatasetSeg(img_file, label_file,128,True)
#
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
# print("number of training images:", len(train_dataset))
#
# test_dataset = UltrasoundDatasetSeg(test_img_file, test_label_file,128,False)
#
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#
# print("number of testing images:", len(test_dataset))

# train_size = int(0.5 * len(dataset))
# test_size = len(dataset) - train_size
#
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print("number of training images:", len(train_dataset))
print("number of testing images:", len(test_dataset))

losses = []
losses_test = []

for epoch in range(start_epoch, num_epoch):
    mean_loss = []
    mean_loss_test = []
    for i, image in enumerate(train_dataloader):
        optimizer.zero_grad()

        x_0 = image[0].to('cuda')
        label = image[1].to('cuda')

        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # # Display the first image
        # axes[0].imshow(x_corrupted.cpu().detach().numpy().squeeze()[0],cmap='gray')
        # axes[0].axis('off')  # Hide the axis

        # # Display the second image
        # axes[1].imshow(ns.cpu().detach().numpy().squeeze()[0],cmap='gray')
        # axes[1].axis('off')  # Hide the axis

        # # Show the plot
        # plt.show()

        # ddd

        x_pred = model(x_0)

        loss = dice_loss(x_pred,label)

        loss.backward()
        optimizer.step()

        mean_loss.append(loss.item())

    scheduler.step()
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(param_group['lr'], min_lr)

    losses.append(np.mean(mean_loss))

    with torch.no_grad():
        for i_test, image_test in enumerate(test_dataloader):
            x_test = image_test[0].to('cuda')
            label_test = image_test[1].to('cuda')

            x_pred_test = model(x_test)

            loss_test = dice_loss(x_pred_test, label_test)

            mean_loss_test.append(loss_test.item())

        losses_test.append(np.mean(mean_loss_test))

    is_best = losses_test[-1] < best_loss
    best_loss = min(losses_test[-1], best_loss)

    save_checkpoint({
        'epoch': epoch,
        'best_loss': best_loss,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best, save_dir, dataset_name)

    print(f"[Epoch {epoch}/{num_epoch}] "
          f"[training loss: {losses[-1]:3f}] "
          f"[test loss: {losses_test[-1]:3f}] "
          f"[best loss: {best_loss:3f}]")

    if epoch % 10 == 0:

        img_save_dir = f"{save_dir}/img_results/{epoch:06}.png"
        if not os.path.exists(os.path.dirname(img_save_dir)):
            os.makedirs(os.path.dirname(img_save_dir))
        x_concat = torch.cat([x_0.view(-1, 1, 128, 128), x_pred.view(-1, 1, 128, 128)], dim=3)
        save_image(x_concat,
                   f"{save_dir}/img_results/{epoch:06}.png",
                   nrow=5, normalize=True)
    # Plotting the loss curve
    plt.plot(losses)  # No label provided, and no legend needed

    # Setting titles and labels
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')

    # Save the plot to a file
    plt.savefig(f"{save_dir}/img_results/loss.png")  # Save the plot as a PNG file
