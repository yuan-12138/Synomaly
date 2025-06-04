"""Forked from https://github.com/Julian-Wyatt/AnoDDPM/blob/3052f0441a472af55d6e8b1028f5d3156f3d6ed3/diffusion_training.py"""
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from tqdm import tqdm

import data_loader
from gaussian_diffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from unet import UNetModel

torch.cuda.empty_cache()


def train(args, resume_model_name=None):
    """
    Train diffusion model.

    :param args: dictionary of parameters
    :param resume_model_name: model checkpoint to resume training, None if starting new training
    :return: trained diffusion model
    """

    training_dataset_loader, synth_anomaly_mask_loader = data_loader.get_training_data(args)

    # show examples of healthy images
    # plt.figure(figsize=(25, 5))
    # for i, image in enumerate(training_dataset_loader):
    #     plt.subplot(1, 5, i+1)
    #     plt.imshow(image.cpu().detach().numpy().squeeze(), cmap="gray")
    #     plt.axis("off")
    # plt.tight_layout()
    # plt.show()

    model = UNetModel(
                args['img_size'], in_channels=args['in_channels'], model_channels=args['model_channels'],
                num_res_blocks=args['num_res_blocks'], attention_resolutions=args['attention_resolutions'],
                dropout=args["dropout"], channel_mult=args['channel_mult'], num_heads=args["num_heads"],
                num_head_channels=args["num_head_channels"],
            )
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    betas = get_beta_schedule(args['beta_schedule'], args['noise_steps'])

    diffusion = GaussianDiffusionModel(
            args['img_size'], betas, img_channels=args['in_channels'], loss_type=args['loss-type'],
            loss_weight=args['loss_weight'], noise_fn=args["noise_fn"], noise_params=args["noise_params"], diffusion_mode="training"
            )

    model_params = None
    if resume_model_name:
        resume_model_path = f'{args["output_path"]}/model/{resume_model_name}'
        model_params = torch.load(resume_model_path)
        model.load_state_dict(model_params["model_state_dict"])
        start_epoch = model_params['n_epoch']
    else:
        start_epoch = 0

    model.to(device)
    optimiser = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['lr_weight_decay'], betas=(0.9, 0.999))
    if resume_model_name:
        optimiser.load_state_dict(model_params["optimizer_state_dict"])
    del model_params

    start_time = time.time()
    losses = []
    best_loss = 1e10

    for epoch in range(start_epoch, args['epochs'] + 1):
        print("Starting epoch ", epoch)
        mean_loss = []

        loaders = zip(training_dataset_loader, synth_anomaly_mask_loader) if synth_anomaly_mask_loader else training_dataset_loader

        for i, data in enumerate(tqdm(loaders)):
            images, synth_anomaly_masks = data if synth_anomaly_mask_loader else (data, None)

            x = images.to(device)
            if synth_anomaly_masks is not None:
                synth_anomaly_masks = synth_anomaly_masks.to(device)

            loss, x_t, estimated_noise = diffusion.p_loss(model, x, synth_anomaly_masks)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimiser.step()

            mean_loss.append(loss.data.cpu())

            if epoch % 100 == 0 and i == 0:
                row_size = min(5, args['batch_size'])
                training_outputs(diffusion, x, synth_anomaly_masks, epoch, row_size, model=model, args=args)

        losses.append(np.mean(mean_loss))
        if epoch % 10 == 0:
            time_taken = time.time() - start_time
            remaining_epochs = args['epochs'] - epoch
            time_per_epoch = time_taken / (epoch + 1 - start_epoch)
            remaining_time = remaining_epochs * time_per_epoch

            print(
                    f"Epoch: {epoch}, "
                    f"mean loss: {losses[-1]:.4f}, "
                    f"last 10 epoch mean loss: {np.mean(losses[-10:]):.4f}, "
                    f"last 50 epoch mean loss: {np.mean(losses[-50:]):.4f}, "
                    f"time per epoch: {int(time_per_epoch // 3600)}h {int((time_per_epoch // 60) % 60)}min, "
                    f"time elapsed: {int(time_taken // 86400)}d {int((time_taken // 3600) % 24)}h {int((time_taken // 60) % 60)}min, "   
                    f"estimated_noise time remaining: {int(remaining_time // 86400)}d {int((remaining_time // 3600) % 24)}h {int((remaining_time // 60) % 60)}min\r"
                    )

        if epoch % 10 == 0 and epoch > 0:
            save_model(unet=model, optimiser=optimiser, args=args, loss=losses[-1], epoch=epoch, status="last")

            plt.plot(range(start_epoch, len(losses) + start_epoch), losses)
            plt.ylim(0, 0.05)
            plt.title("Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(f'{args["output_path"]}/diffusion-training-images/{args["json_file_name"]}/loss_curve.png')

        if np.mean(losses[-10:]) < best_loss:
            best_loss = np.mean(losses[-10:])
            save_model(unet=model, optimiser=optimiser, args=args, loss=losses[-1], epoch=epoch, status="best")


def save_model(unet, optimiser, args, loss=0, epoch=0, status="last"):
    """
    Save model.

    :param unet: unet instance
    :param optimiser: ADAM optim
    :param args: model parameters
    :param loss: loss for checkpoint
    :param epoch: epoch for checkpoint
    :param status: last model or best model
    :return: saved model
    """

    if status == "last":
        model_path = f'{args["output_path"]}/model/{args["json_file_name"]}/last_model.pt'
        torch.save(
                {
                    'n_epoch':              args["epochs"],
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    "args":                 args,
                    'loss':                 loss,
                    }, model_path
                )
        print(f"Last model saved as {model_path}. Epoch {epoch}")
    else:
        model_path = f'{args["output_path"]}/model/{args["json_file_name"]}/best_model.pt'
        torch.save(
                {
                    'n_epoch':              epoch,
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    "args":                 args,
                    'loss':                 loss,
                    }, model_path
                )
        print(f"Best model saved as {model_path}. Epoch {epoch}")


def training_outputs(diffusion, x, synth_anomaly_masks, epoch, row_size, model, args):
    """
    Performs diffusion process with current model and saves current results.

    :param diffusion: diffusion model instance
    :param x: input image
    :param synth_anomaly_masks: mask for synthetic anomaly regions
    :param epoch: trained epochs
    :param row_size: rows for outputs
    :param model: diffusion model for sampling
    :param args: sampling parameters
    :return: diffusion results with current model
    """

    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
    noise = diffusion.noise_fn(x, synth_anomaly_masks)
    x_t = diffusion.q_sample(x, t, noise)
    x_pred = diffusion.p_sample(model, x_t, t)

    # id = 0
    # difference = (x - x_pred)**2
    # plt.figure(figsize=(20, 10))
    # plt.subplot(2, 5, 1)
    # plt.imshow(x.cpu()[id, 0], cmap='gray')
    # plt.title('Input image')
    # plt.subplot(2, 5, 2)
    # plt.imshow(x_t.cpu().detach().numpy()[id, 0], cmap='gray', vmin=-1, vmax=1)
    # plt.title('Noised image')
    # plt.subplot(2, 5, 3)
    # plt.imshow(x_pred.cpu().detach().numpy()[id, 0], cmap='gray', vmin=-1, vmax=1)
    # plt.title('Generated healthy image')
    # plt.subplot(2, 5, 4)
    # plt.imshow(noise.cpu().detach().numpy()[id, 0], cmap='gray')
    # plt.title('True noise')
    # plt.subplot(2, 5, 5)
    # plt.imshow(difference.cpu().detach().numpy()[id, 0], cmap='gray')
    # plt.title('Detected anomaly')
    #
    # plt.subplot(2, 5, 6)
    # plt.hist(x.cpu()[id, 0].flatten())
    # plt.subplot(2, 5, 7)
    # plt.hist(x_t.cpu().detach().numpy()[id, 0].flatten())
    # plt.subplot(2, 5, 8)
    # plt.hist(x_pred.cpu().detach().numpy()[id, 0].flatten())
    # plt.subplot(2, 5, 9)
    # plt.hist(noise.cpu().detach().numpy()[id, 0].flatten())
    # plt.subplot(2, 5, 10)
    # plt.hist(difference.cpu().detach().numpy()[id, 0].flatten())
    # plt.suptitle(f'{t[id]} timesteps')
    # plt.tight_layout()
    # plt.show()

    out = torch.cat(
            (x[:row_size, ...].cpu(),
             x_t[:row_size, ...].cpu(),
             x_pred[:row_size, ...].cpu())
            )
    plt.figure()
    plt.xticks(np.arange(int(args['img_size']/2), int(row_size*(args['img_size']+2)+args['img_size']/2), args['img_size']+2), t.cpu().numpy()[:row_size])
    plt.yticks(np.arange(int(args['img_size']/2), int(3*(args['img_size']+2)+args['img_size']/2), args['img_size']+2), ['input', 'noised', 'pred'])
    plt.xlabel('noise steps')
    plt.title(f'Input, noised and predicted image - Epoch {epoch}')
    plt.rcParams['figure.dpi'] = 150
    plt.grid(False)
    plt.imshow(gridify_output(out, row_size), cmap='gray')

    image_path = f'{args["output_path"]}/diffusion-training-images/{args["json_file_name"]}/epoch={epoch}.png'
    plt.savefig(image_path)
    print(f"Sample images saved as {image_path}")
    plt.close('all')


def launch_training(server, resume_model_name=None):
    """
    Launch training

    :param server: boolean for using server
    :param resume_model_name: model checkpoint to resume training, None if starting new training
    """

    if len(sys.argv[1:]) > 0:
        json_file_name = sys.argv[1]
    else:
        json_file_name = "args_us_47"

    args = get_args_from_json(json_file_name, server)
    print(args["apply_mask"])

    # create output directories
    for i in ['model', 'diffusion-training-images']:
        try:
            os.makedirs(os.path.join(args['output_path'], i), exist_ok=True)
            os.makedirs(os.path.join(args['output_path'], i, json_file_name), exist_ok=True)
        except OSError:
            pass
    os.makedirs(os.path.join(args['output_path'], 'model', json_file_name), exist_ok=True)

    print(args)

    train(args, resume_model_name)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    using_server = 'None'  # ['IFL', 'TranslaTUM', 'None']
    resume_training_model = None  # ['args_us_XX/best_model.pt', None]

    launch_training(using_server, resume_training_model)

