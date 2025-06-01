from unet import UNetModel
import data_loader
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
import random
import numpy as np
import os
import shutil
from torchvision.utils import save_image
import itertools
import cv2
from scipy.ndimage import label
import warnings
from sklearn.metrics import precision_recall_curve, auc
import time
from gaussian_diffusion import GaussianDiffusionModel, get_beta_schedule
import lpips
from torch.cuda.amp import autocast

def write_log_file(file_path,txt):
    with open(file_path, 'a') as f:
        f.write(txt + '\n')

def create_gaussian_blur_difference_map(x_0, x_pred, kernel_size=3, threshold=5.0):
    x_0_array = x_0.cpu().squeeze().numpy()
    x_0_blurred = cv2.GaussianBlur(x_0_array, (kernel_size, kernel_size), 0)
    x_pred_array = x_pred.cpu().detach().numpy().squeeze()
    x_pred_blurred = cv2.GaussianBlur(x_pred_array, (kernel_size, kernel_size), 0)

    diff = abs(x_0_blurred - x_pred_blurred)
    diff[diff < threshold] = 0

    diff_final = remove_small_spots(diff)
    return diff_final

def remove_small_spots(map, threshold=30):
    binary_map = map > 0
    labeled_map, num_features = label(binary_map)
    component_sizes = np.bincount(labeled_map.ravel())
    large_components_masked = component_sizes[labeled_map] >= threshold
    return large_components_masked * map

def get_dice_score(diff_truth, diff_pred):
    if diff_truth.sum() == 0 and diff_pred.sum() == 0:
        return 1.0
    dice_score = 2 * (diff_truth & diff_pred).sum() / (diff_truth.sum() + diff_pred.sum())
    return round(dice_score, 4)


def get_iou_score(diff_truth, diff_pred):
    if diff_truth.sum() == 0 and diff_pred.sum() == 0:
        return 1.0
    intersection = np.logical_and(diff_truth, diff_pred)
    union = np.logical_or(diff_truth, diff_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return round(iou_score, 4)

def get_precision_score(diff_truth, diff_pred):
    if diff_truth.sum() == 0 and diff_pred.sum() == 0:
        return 1.0
    true_positives = np.sum(diff_truth & diff_pred)
    false_positives = np.sum(diff_pred) - true_positives
    if true_positives + false_positives != 0:
        precision_score = true_positives / (true_positives + false_positives)
    else:
        precision_score = 0.0
    return round(precision_score, 4)


def get_recall_score(diff_truth, diff_pred):
    if diff_truth.sum() == 0 and diff_pred.sum() == 0:
        return 1.0
    true_positives = np.sum(diff_truth & diff_pred)
    false_negatives = np.sum(diff_truth) - true_positives
    if true_positives + false_negatives != 0:
        recall_score = true_positives / (true_positives + false_negatives)
    else:
        recall_score = 0.0
    return round(recall_score, 4)


def get_fpr_score(diff_truth, diff_pred):
    false_positives = np.sum(diff_pred) - np.sum(diff_truth & diff_pred)
    true_negatives = np.sum(~diff_truth & ~diff_pred)
    try:
        fpr_score = false_positives / (false_positives + true_negatives)
    except:
        fpr_score = 0.0
    return round(fpr_score, 4)


def get_auprc_score(diff_truth, diff_pred):
    warnings.filterwarnings("ignore", message="No positive class found in y_true*")
    precision, recall, thresholds = precision_recall_curve(diff_truth.flatten(), diff_pred.flatten())
    auprc = auc(recall, precision)
    return round(auprc, 4)


def get_hausdorff_distance(diff_truth, diff_pred, percentile=95):
    from scipy.spatial import distance
    # Compute all pairwise distances
    dists = distance.cdist(diff_truth, diff_pred, 'euclidean')

    # Compute directed Hausdorff distances
    distance_1 = np.min(dists, axis=1)
    distance_2 = np.min(dists, axis=0)

    # Find the percentile distance
    hd1 = np.percentile(distance_1, percentile)
    hd2 = np.percentile(distance_2, percentile)

    return round(max(hd1, hd2), 4)


def get_ssim_score(x_0, x_pred):
    from skimage.metrics import structural_similarity as ssim
    x_0_array = x_0.cpu().detach().numpy().squeeze()
    x_pred_array = x_pred.cpu().detach().numpy().squeeze()
    ssim_score = ssim(x_0_array, x_pred_array, data_range=x_0_array.max() - x_0_array.min())
    return round(ssim_score, 4)

def lpips_loss(l_pips_sq, anomaly_img, ph_img, retPerLayer=False):
    """
    :param anomaly_img: anomaly image
    :param ph_img: pseudo-healthy image
    :param retPerLayer: whether to return the loss per layer
    :return: LPIPS loss
    """
    if len(ph_img.shape) < 2:
        print('Image should have 2 dimensions at lease (LPIPS)')
        return
    if len(ph_img.shape) == 2:
        ph_img = torch.unsqueeze(torch.unsqueeze(ph_img, 0), 0)
        anomaly_img = torch.unsqueeze(torch.unsqueeze(anomaly_img, 0), 0)
    if len(ph_img.shape) == 3:
        ph_img = torch.unsqueeze(ph_img, 0)
        anomaly_img = torch.unsqueeze(anomaly_img, 0)

    saliency_maps = []
    for batch_id in range(anomaly_img.size(0)):
        lpips = l_pips_sq(2*anomaly_img[batch_id:batch_id + 1, :, :, :]-1, 2*ph_img[batch_id:batch_id + 1, :, :, :]-1,
                               normalize=True, retPerLayer=retPerLayer)
        if retPerLayer:
            lpips = lpips[1][0]
        saliency_maps.append(lpips[0,:,:,:].cpu().detach().numpy())
    return np.asarray(saliency_maps)

def dilate_masks(masks):
    """
    :param masks: masks to dilate
    :return: dilated masks
    """
    kernel = np.ones((3, 3), np.uint8)

    dilated_masks = torch.zeros_like(masks)
    for i in range(masks.shape[0]):
        mask = masks[i][0].detach().cpu().numpy()
        if np.sum(mask) < 1:
            dilated_masks[i] = masks[i]
            continue
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        dilated_mask = torch.from_numpy(dilated_mask).to(masks.device).unsqueeze(dim=0)
        dilated_masks[i] = dilated_mask

    return dilated_masks


def run_inference(l_pips_sq, test_dataset_loader, model, diffusion, groundtruth_anomaly_masks, inference_noise_steps, gaussian_blur_kernel_size, anomaly_threshold, noise_level_inpaint=50,
                  resample_steps=4,device='cuda'):
    evaluation_score_list = []
    process_time_list = []
    for i, image in enumerate(test_dataset_loader):
        x_0 = image.to(device)

        process_start_time = time.time()

        with torch.no_grad():
            t = torch.tensor([inference_noise_steps], device=x_0.device).repeat(x_0.shape[0])
            noise = diffusion.noise_fn(x_0, None)
            x_t = diffusion.q_sample(x_0, t, noise)
            x_rec = diffusion.p_sample(model, x_t, t)

            # x_rec = torch.clamp(x_rec, 0, 1)
            # x_res = self.ano_map.compute_residual(inputs, x_rec, hist_eq=False)
            x_res = np.abs(x_rec.cpu().detach().numpy() - x_0.cpu().detach().numpy())
            lpips_mask = lpips_loss(l_pips_sq, x_0, x_rec, retPerLayer=False)
            #
            # anomalous: high value, healthy: low value
            x_res = np.asarray([(x_res[i] / np.percentile(x_res[i], 95)) for i in range(x_res.shape[0])]).clip(0, 1)
            combined_mask_np = lpips_mask * x_res
            combined_mask = torch.Tensor(combined_mask_np).to(device)
            masking_threshold = torch.tensor(np.asarray([(
                np.percentile(combined_mask[i].cpu().detach().numpy(), 95)) for i in
                range(combined_mask.shape[0])]).clip(0,
                                                     1))
            combined_mask_binary = torch.cat([torch.where(combined_mask[i] > masking_threshold[i], torch.ones_like(
                torch.unsqueeze(combined_mask[i], 0)), torch.zeros_like(combined_mask[i]))
                                              for i in range(combined_mask.shape[0])], dim=0)

            combined_mask_binary_dilated = dilate_masks(combined_mask_binary)
            mask_in_use = combined_mask_binary_dilated

            x_masked = (1 - mask_in_use) * x_0
            x_rec_masked = mask_in_use * x_rec
            #
            #
            # 2. Start in-painting with reconstructed image and not pure noise

            timesteps = torch.tensor([noise_level_inpaint], device=x_rec.device).repeat(x_rec.shape[0])
            noise = diffusion.noise_fn(x_rec, None)
            inpaint_image = diffusion.q_sample(x_rec, timesteps, noise)


            # 3. Setup for loop
            # timesteps = self.inference_scheduler.get_timesteps(noise_level_inpaint)
            timesteps = torch.from_numpy(np.arange(0, noise_level_inpaint + 1)[::-1].copy())
            progress_bar = iter(timesteps)
            num_resample_steps = resample_steps
            stitched_images = []

            # 4. Inpainting loop
            with torch.no_grad():
                with autocast(enabled=True):
                    for t in progress_bar:
                        for u in range(num_resample_steps):
                            # 4a) Get the known portion at t-1
                            if t > 0:
                                # noise = torch.randn_like(x_0, device=device)
                                # timesteps_prev = torch.full([x_0.shape[0]], t - 1, device=device).long()
                                # noised_masked_original_context = self.inference_scheduler.add_noise(
                                #     original_samples=x_masked, noise=noise, timesteps=timesteps_prev
                                # )
                                timesteps_prev = torch.tensor([t - 1], device=x_0.device).repeat(x_0.shape[0])
                                noise = diffusion.noise_fn(x_0, None)
                                noised_masked_original_context = diffusion.q_sample(x_masked, timesteps_prev, noise)
                            else:
                                noised_masked_original_context = x_masked
                            #
                            # 4b) Perform a denoising step to get the unknown portion at t-1
                            if t > 0:
                                # timesteps = torch.full([inputs.shape[0]], t, device=self.device).long()
                                # model_output = self.unet(x=inpaint_image, timesteps=timesteps)
                                # inpainted_from_x_rec, _ = self.inference_scheduler.step(model_output, t,
                                #                                                         inpaint_image)
                                timesteps = torch.tensor([t], device=inpaint_image.device).repeat(inpaint_image.shape[0])
                                out = diffusion.p_mean_variance(model, inpaint_image, timesteps)
                                inpainted_from_x_rec = out["pred_x_t_1"]

                            #
                            # 4c) Combine the known and unknown portions at t-1
                            inpaint_image = torch.where(
                                mask_in_use == 1, inpainted_from_x_rec, noised_masked_original_context
                            )

                            ## 4d) Perform resampling: sample x_t from x_t-1 -> get new image to be inpainted
                            # in the masked region
                            if t > 0 and u < (num_resample_steps - 1):
                                inpaint_image = (
                                        torch.sqrt(1 - torch.tensor(diffusion.betas[t - 1]).to(device)) * inpaint_image
                                        + torch.sqrt(torch.tensor(diffusion.betas[t - 1]).to(device))
                                        * torch.randn_like(x_0, device=device)
                                )

            final_inpainted_image = inpaint_image
            x_res_2 = np.abs(final_inpainted_image.cpu().detach().numpy() - x_0.cpu().detach().numpy())
            x_lpips_2 = lpips_loss(l_pips_sq, x_0, final_inpainted_image, retPerLayer=False)
            anomaly_maps = x_res_2 * combined_mask.cpu().detach().numpy()
            anomaly_scores = np.mean(anomaly_maps, axis=(1, 2, 3), keepdims=True)

        x_pred = final_inpainted_image
        difference_map = create_gaussian_blur_difference_map(x_0, final_inpainted_image, kernel_size=gaussian_blur_kernel_size,
                                                             threshold=anomaly_threshold)

        process_end_time = time.time()
        process_time = process_end_time - process_start_time
        process_time_list.append(process_time)

        print('Average Calculation Time: ', np.array(process_time_list).mean())

        groundtruth_anomaly_mask = groundtruth_anomaly_masks[i]
        groundtruth_anomaly = groundtruth_anomaly_mask.astype(bool)
        predicted_anomaly = difference_map.astype(bool)

        evaluation_scores = {
            "dice_score": get_dice_score(groundtruth_anomaly, predicted_anomaly),
            "auprc_score": get_auprc_score(groundtruth_anomaly, predicted_anomaly),
            "iou_score": get_iou_score(groundtruth_anomaly, predicted_anomaly),
            "precision_score": get_precision_score(groundtruth_anomaly, predicted_anomaly),
            "recall_score": get_recall_score(groundtruth_anomaly, predicted_anomaly),
            "fpr_score": get_fpr_score(groundtruth_anomaly, predicted_anomaly),
            "hausdorff_score": get_hausdorff_distance(groundtruth_anomaly, predicted_anomaly),
            "ssim_score": get_ssim_score(x_0, x_pred)
        }
        evaluation_score_list.append(evaluation_scores)

    score_names = list(evaluation_score_list[0].keys())
    evaluation_mean = {score_name: round(np.mean([entry[score_name] for entry in evaluation_score_list]), 4) for
                       score_name in score_names}
    evaluation_std = {score_name: round(np.std([entry[score_name] for entry in evaluation_score_list]), 4) for
                      score_name in score_names}

    print(
        f"Average DICE score: {evaluation_mean['dice_score']}+-{evaluation_std['dice_score']}, average AUPRC score: {evaluation_mean['auprc_score']}+-{evaluation_std['auprc_score']}, average IOU score: {evaluation_mean['iou_score']}+-{evaluation_std['iou_score']}, average Precision: {evaluation_mean['precision_score']}+-{evaluation_std['precision_score']}, average Recall: {evaluation_mean['recall_score']}+-{evaluation_std['recall_score']}, average FPR: {evaluation_mean['fpr_score']}+-{evaluation_std['fpr_score']}, average Hausdorff: {evaluation_mean['hausdorff_score']}+-{evaluation_std['hausdorff_score']}, average SSIM: {evaluation_mean['ssim_score']}+-{evaluation_std['ssim_score']}")
    write_log_file(log_path,
                   f"Average DICE score: {evaluation_mean['dice_score']}+-{evaluation_std['dice_score']}, average AUPRC score: {evaluation_mean['auprc_score']}+-{evaluation_std['auprc_score']}, average IOU score: {evaluation_mean['iou_score']}+-{evaluation_std['iou_score']}, average Precision: {evaluation_mean['precision_score']}+-{evaluation_std['precision_score']}, average Recall: {evaluation_mean['recall_score']}+-{evaluation_std['recall_score']}, average FPR: {evaluation_mean['fpr_score']}+-{evaluation_std['fpr_score']}, average Hausdorff: {evaluation_mean['hausdorff_score']}+-{evaluation_std['hausdorff_score']}, average SSIM: {evaluation_mean['ssim_score']}+-{evaluation_std['ssim_score']}")

    return evaluation_mean, evaluation_std





dataset_name = 'brats23'
pretrained_path = '../output/BraTS/model/args_brats_11/best_model.pt'
mode = "anomalous"
log_path = os.path.join(os.path.dirname(pretrained_path),'log_autoDDPM'+mode+'.txt')

inference_noise_steps_list = [200]
gaussian_blur_kernel_size_list = [9]
anomaly_threshold_list = [0.1]
# pretrained_path = None

betas = get_beta_schedule("cosine", 1000)

diffusion = GaussianDiffusionModel(
            128, betas, img_channels=1, loss_type="vlb",
            loss_weight="none", noise_fn="gaussian", noise_params=None, diffusion_mode="inference"
            )

model = UNetModel(128, in_channels=1, model_channels=128,
                num_res_blocks=2, attention_resolutions="32,16,8",
                dropout=0.0, channel_mult="", num_heads=2,
                num_head_channels=64,).to('cuda')
checkpoint = torch.load(pretrained_path)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

test_dataset_loader, groundtruth_anomaly_masks = data_loader.get_test_data_(dataset_name=dataset_name,mode=mode,num_test_img=2000)

combination_scores = []
parameters_ranges = [inference_noise_steps_list, gaussian_blur_kernel_size_list, anomaly_threshold_list]
print(f"Testing on inference noise steps: {inference_noise_steps_list} gaussian blur kernel sizes: {gaussian_blur_kernel_size_list} anomaly thresholds: {anomaly_threshold_list}")

l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True,spatial=True, lpips=True).to('cuda')

# Test on all combinations of parameters
for combination in itertools.product(*parameters_ranges):
    inference_noise_steps = combination[0]
    gaussian_blur_kernel_size = combination[1]
    anomaly_threshold = combination[2]
    if (inference_noise_steps == 100 and gaussian_blur_kernel_size == 3) or (inference_noise_steps == 100 and gaussian_blur_kernel_size == 5):
        continue
    print(f"Noise steps: {inference_noise_steps}, kernel size: {gaussian_blur_kernel_size}, threshold: {anomaly_threshold}")
    write_log_file(log_path,f"Noise steps: {inference_noise_steps}, kernel size: {gaussian_blur_kernel_size}, threshold: {anomaly_threshold}")

    # Run inference on the test dataset with the given parameters
    evaluation_mean, evaluation_std = run_inference(l_pips_sq, test_dataset_loader, model, diffusion,
                                                    groundtruth_anomaly_masks, inference_noise_steps,
                                                    gaussian_blur_kernel_size,anomaly_threshold)

    # Store the evaluation scores
    combination_scores.append((combination, evaluation_mean, evaluation_std))

sorted_dice_scores = sorted(combination_scores, key=lambda x: x[1]['dice_score'], reverse=True)
print(f"Best combinations:")
write_log_file(log_path,f"Best combinations:")

for i, (combination, evaluation_mean, evaluation_std) in enumerate(sorted_dice_scores[:20], 1):
    print(f"#{i} Combination: {combination} Dice score: {evaluation_mean['dice_score']}+-{evaluation_std['dice_score']}, AUPRC score: {evaluation_mean['auprc_score']}+-{evaluation_std['auprc_score']}, IoU score: {evaluation_mean['iou_score']}+-{evaluation_std['iou_score']}, Hausdorff score: {evaluation_mean['hausdorff_score']}+-{evaluation_std['hausdorff_score']}")
    write_log_file(log_path,f"#{i} Combination: {combination} Dice score: {evaluation_mean['dice_score']}+-{evaluation_std['dice_score']}, AUPRC score: {evaluation_mean['auprc_score']}+-{evaluation_std['auprc_score']}, IoU score: {evaluation_mean['iou_score']}+-{evaluation_std['iou_score']}, Hausdorff score: {evaluation_mean['hausdorff_score']}+-{evaluation_std['hausdorff_score']}")
