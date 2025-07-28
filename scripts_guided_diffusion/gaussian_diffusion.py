"""Forked from https://github.com/Julian-Wyatt/AnoDDPM/blob/3052f0441a472af55d6e8b1028f5d3156f3d6ed3/GaussianDiffusion.py"""

import random

import cv2
import numpy as np
import torch as th
import skimage.exposure

from nn import mean_flat
from losses import normal_kl, discretized_gaussian_log_likelihood
from torch.nn import functional as F
from helpers import *
from simplex import Simplex_CLASS
from torchvision import transforms


def get_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al., extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        max_beta = 0.999
        f = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - f(t2) / f(t1), max_beta))
        return np.array(betas)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def extract(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    res = res[:broadcast_shape[0]]
    return res.expand(broadcast_shape)


# def approx_standard_normal_cdf(x):
#     """
#     A fast approximation of the cumulative distribution function of the
#     standard normal.
#     """
#     return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def generate_gaussian_noise(x):
    """
    Generate Gaussian noise.

    :param x: input image
    :return: gaussian noise
    """

    noise = torch.randn_like(x)
    return noise

def generate_pyramid_noise(x,discount=0.8):
    u = transforms.Resize(x.shape[2], antialias=True)
    noise = torch.randn_like(x)
    w = x.shape[2]
    h = x.shape[3]
    for i in range(10):
        r = random.random() * 2 + 2  # Rather than always going 2x,
        w, h = max(1, int(w / (r ** i))), max(1, int(h / (r ** i)))
        noise += u(torch.randn_like(x)) * discount ** i
        if w == 1 or h == 1:
            break  # Lowest resolution is 1x1
    return noise / noise.std()  # Scaled back to roughly unit variance



def generate_simplex_noise(simplex_instance, x, octave=6, persistence=0.8, frequency=64):
    # Forked from https://github.com/Julian-Wyatt/AnoDDPM/blob/3052f0441a472af55d6e8b1028f5d3156f3d6ed3/GaussianDiffusion.py
    """
    Generate simplex noise.

    :param simplex_instance: simplex instance
    :param x: input image
    :param octave: octave
    :param persistence: persistence
    :param frequency: frequency
    :return: simplex noise
    """

    simplex_instance.newSeed()
    noise = torch.from_numpy(simplex_instance.rand_3d_octaves(x.shape, octave, persistence, frequency)).to(x.device)
    return noise


def generate_coarse_noise(x, noise_res=16, noise_std=0.2):
    # Forked from https://github.com/AntanasKascenas/DenoisingAE/blob/8e5a9df1704fd153887943923780839620492442/src/denoising.py#L126
    """
    Generate coarse noise.

    :param x: input image
    :param noise_res: noise resolution
    :param noise_std: noise standard deviation
    :return: coarse noise
    """

    noise = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], noise_res, noise_res), std=noise_std).to(x.device)
    noise = F.upsample_bilinear(noise, size=[x.shape[2], x.shape[3]])

    # Roll to randomly translate the generated noise.
    roll_x = random.choice(range(x.shape[2]))
    roll_y = random.choice(range(x.shape[3]))
    noise = torch.roll(noise, shifts=[roll_x, roll_y], dims=[-2, -1])
    return noise


def generate_synomaly_noise(x, synth_anomaly_masks, mode, anomaly_sigma=7, anomaly_threshold=150, anomaly_offset=0.5, anomaly_direction=1):
    """
    Generate Synomaly noise.

    :param x: input image
    :param synth_anomaly_masks: masks for synthetic noise regions
    :param mode: training (add synthetic anomalies) or inference (not add synthetic anomalies)
    :param anomaly_sigma: size of anomalies, the larger the sigma, the larger the anomalies, brats 3, ultrasound 7, lits 5
    :param anomaly_threshold: threshold for anomalies, the larger the threshold, the fewer anomalies, brats 175, ultrasound 150, lits 175
    :param anomaly_offset: offset for anomalies, the larger the offset, the brighter the anomalies, brats 0.5, ultrasound 0.5, lits 0.75
    :param anomaly_direction: direction of anomalies, 1 for brighter, -1 for darker, brats 1, ultrasound 1, lits -1
    :return: Synomaly noise
    """

    noise = torch.zeros(x.shape)
    height, width = x.shape[2:4]

    for i in range(x.shape[0]):
        # create Gaussian background noise
        background_noise = np.random.randn(height, width)

        if mode == "inference":  # do not add synthetic anomalies in inference
            noise[i, 0] = torch.from_numpy(background_noise)

        else:
            # create a mask for shapes
            blur = cv2.GaussianBlur(background_noise, (0, 0), sigmaX=anomaly_sigma, sigmaY=anomaly_sigma, borderType=cv2.BORDER_DEFAULT)
            stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255))
            shape_mask = cv2.threshold(stretch, anomaly_threshold, 1, cv2.THRESH_BINARY)[1]

            # crop the mask only to relevant area
            if synth_anomaly_masks is not None:
                synth_anomaly_mask = synth_anomaly_masks[i, 0].cpu().numpy()   # -1 regions not add synthetic anomaly, 1 regions add synthetic anomaly
                shape_mask[synth_anomaly_mask == -1] = 0
            # combine background noise and shape noise
            combined = background_noise.copy()
            if isinstance(anomaly_direction, list):
                anomaly_direction_ = random.choice(anomaly_direction)
            else:
                anomaly_direction_ = anomaly_direction
            combined[shape_mask == 1] += anomaly_direction_ * (torch.rand(1).item() + anomaly_offset)  # add or reduce 0.0-1.0 plus offset

            noise[i, 0] = torch.from_numpy(combined)

    return noise.to(x.device)


def generate_noise(noise_fn, noise_params, mode):
    if noise_fn == "gaussian":
        return lambda x, _: generate_gaussian_noise(x)
    elif noise_fn == "synomaly":
        return lambda x, synth_anomaly_masks: generate_synomaly_noise(x, synth_anomaly_masks, mode, **noise_params)
    elif noise_fn == "coarse":
        return lambda x, _: generate_coarse_noise(x)
    elif noise_fn == "simplex":
        simplex = Simplex_CLASS()
        return lambda x, _: generate_simplex_noise(simplex, x)
    elif noise_fn == "pyramid":
        return lambda x, _: generate_pyramid_noise(x)
    else:
        raise NotImplementedError(f"unknown noise function: {noise_fn}")


class GaussianDiffusionModel:
    def __init__(
            self,
            img_size,
            betas,
            img_channels=1,
            loss_type="l2",  # l1 / l2 / vlb
            loss_weight="none",  # 'prop t', 'uniform', 'none'
            noise_fn="gaussian",  # gaussian / coarse / simplex
            noise_params=None,   # None / {anomaly_radius, anomaly_sigma, anomaly_threshold, anomaly_offset}
            diffusion_mode="training",  # training / inference
    ):
        super().__init__()

        self.img_size = img_size
        self.img_channels = img_channels
        self.loss_type = loss_type
        self.loss_weight = loss_weight

        self.noise_fn = generate_noise(noise_fn, noise_params, diffusion_mode)

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        self.num_timesteps = len(betas)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        if loss_weight == 'prop-t':
            self.weights = np.arange(self.num_timesteps, 0, -1)
        elif loss_weight == "uniform":
            self.weights = np.ones(self.num_timesteps)

        alphas = 1.0 - betas
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_betas = np.sqrt(betas)

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def sample_t_with_weights(self, b_size, device):
        p = self.weights / np.sum(self.weights)
        indices_np = np.random.choice(len(p), size=b_size, p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / len(p) * p[indices_np]
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights

    def predict_x_0_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps)

    def predict_eps_from_x_0(self, x_t, t, pred_x_0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_x_0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_mean_variance(self, x_0, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_0: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_0's shape.
        """
        mean = (extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0)
        variance = extract(1.0 - self.alphas_cumprod, t, x_0.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_0.shape)
        return mean, variance, log_variance

    def q_sample(self, x_0, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_0: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: generated noise from noise_fn, if None take Gaussian noise.
        :return: A noisy version of x_0.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape
        return (extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise)

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        # mu (x_t,x_0) = \frac{\sqrt{alphacumprod prev} betas}{1-alphacumprod} *x_0
        # + \frac{\sqrt{alphas}(1-alphacumprod prev)}{ 1- alphacumprod} * x_t
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
                          + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)

        # var = \frac{1-alphacumprod prev}{1-alphacumprod} * betas
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_0.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x_t, t, estimate_noise=None):
        """
        Finds the mean & variance from N(x_{t-1}; mu_theta(x_t,t), sigma_theta (x_t,t))
        """
        if estimate_noise is None:
            estimate_noise = model(x_t, t)

        # fixed model variance defined as \hat{\beta}_t - could add learned parameter
        model_variance = np.append(self.posterior_variance[1], self.betas[1:])
        model_log_variance = np.log(model_variance)
        model_variance = extract(model_variance, t, x_t.shape)
        model_log_variance = extract(model_log_variance, t, x_t.shape)

        pred_x_0 = self.predict_x_0_from_eps(x_t, t, eps=estimate_noise).clamp(-1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(pred_x_0, x_t, t)

        pred_x_t_1 = model_mean + (model_variance ** 0.5) * self.noise_fn(estimate_noise, None)
        assert (model_mean.shape == model_log_variance.shape == pred_x_0.shape == x_t.shape)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_x_0": pred_x_0,
            "pred_x_t_1": pred_x_t_1
        }

    def p_sample(self, model, x_t, t):
        """
        Sample x_{t-1} from the model at the given timestep.
        """
        out = self.p_mean_variance(model, x_t, t)
        # noise = torch.randn_like(x_t)
        # if type(denoise_fn) == str:
        #     if denoise_fn == "gauss":
        #         noise = torch.randn_like(x_t)
        #     elif denoise_fn == "noise_fn":
        #         noise = self.noise_fn(x_t, t).float()
        #     elif denoise_fn == "random":
        #         # noise = random_noise(self.simplex, x_t, t).float()
        #         noise = torch.randn_like(x_t)
        #     else:
        #         noise = generate_simplex_noise(self.simplex, x_t, t, False, in_channels=self.img_channels).float()
        # else:
        #     noise = self.noise_fn(x_t, t).float()

        # noise = self.noise_fn(x_t, t).float()
        #
        # nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))  # no noise when t == 0
        # sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        # return {"sample": sample, "pred_x_0": out["pred_x_0"]}
        return out["pred_x_0"]


    # def forward_backward(self, model, x, see_whole_sequence="half", t_distance=None):
    #     """Only used for videos. Not checked."""
    #     assert see_whole_sequence == "whole" or see_whole_sequence == "half" or see_whole_sequence is None
    #
    #     if t_distance == 0:
    #         return x.detach()
    #     if t_distance is None:
    #         t_distance = self.num_timesteps
    #     seq = [x.cpu().detach()]
    #
    #     if see_whole_sequence == "whole":
    #         for t in range(int(t_distance)):
    #             t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
    #             # noise = torch.randn_like(x)
    #             noise = self.noise_fn(x, t_batch).float()
    #             with torch.no_grad():
    #                 x = self.sample_q_gradual(x, t_batch, noise)
    #
    #             seq.append(x.cpu().detach())
    #     else:
    #         # x = self.sample_q(x,torch.tensor([t_distance], device=x.device).repeat(x.shape[0]),torch.randn_like(x))
    #         t_tensor = torch.tensor([t_distance - 1], device=x.device).repeat(x.shape[0])
    #         x = self.q_sample(
    #             x, t_tensor,
    #             self.noise_fn(x, t_tensor).float()
    #         )
    #         if see_whole_sequence == "half":
    #             seq.append(x.cpu().detach())
    #
    #     for t in range(int(t_distance) - 1, -1, -1):
    #         t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
    #         with torch.no_grad():
    #             out = self.p_sample(model, x, t_batch)
    #             x = out["sample"]
    #         if see_whole_sequence:
    #             seq.append(x.cpu().detach())
    #
    #     return x.detach() if not see_whole_sequence else seq

    # def sample_q_gradual(self, x_t, t, noise):
    #     """
    #     Only used for videos. Not checked.
    #     :param x_t:
    #     :param t:
    #     :param noise:
    #     :return:
    #     """
    #     return (extract(self.sqrt_alphas, t, x_t.shape) * x_t +
    #             extract(self.sqrt_betas, t, x_t.shape) * noise)

    def calc_vlb_xt(self, model, x_0, x_t, t, estimate_noise=None):
        """
        Get a term for the variational lower-bound. Find KL divergence at t.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_x_0': the x_0 predictions.
        """
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_0, x_t, t)
        out = self.p_mean_variance(model, x_t, t, estimate_noise)
        kl = normal_kl(true_mean, true_log_var, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_0, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_0.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_x_0": out["pred_x_0"]}

    def calc_loss(self, model, x_0, synth_anomaly_masks, t):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_0: the [N x C x ...] tensor of inputs.
        :param synth_anomaly_masks: masks for synthetic noise regions
        :param t: a batch of timestep indices.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        # true_noise = torch.randn_like(x)
        true_noise = self.noise_fn(x_0, synth_anomaly_masks).float()

        x_t = self.q_sample(x_0, t, true_noise)
        estimated_noise = model(x_t, t)
        loss = {}
        if self.loss_type == "l1":
            loss["loss"] = mean_flat((estimated_noise - true_noise).abs())
        elif self.loss_type == "l2":
            loss["loss"] = mean_flat((estimated_noise - true_noise).square())
        elif self.loss_type == "vlb":
            # add vlb term
            loss["vlb"] = self.calc_vlb_xt(model, x_0, x_t, t, estimated_noise)["output"]
            loss["loss"] = loss["vlb"] + mean_flat((estimated_noise - true_noise).square())
        else:
            loss["loss"] = mean_flat((estimated_noise - true_noise).square())
        return loss, x_t, estimated_noise

    def p_loss(self, model, x_0, synth_anomaly_masks):
        if self.loss_weight == "none":
            # if args["train_start"]:
            #     t = torch.randint(0, min(args["sample_distance"], self.num_timesteps), (x_0.shape[0],), device=x_0.device)
            # else:
            #     t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device)
            t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device)
            weights = 1
        else:
            t, weights = self.sample_t_with_weights(x_0.shape[0], x_0.device)

        loss, x_t, estimated_noise = self.calc_loss(model, x_0, synth_anomaly_masks, t)
        mean_loss = (loss["loss"] * weights).mean()
        return mean_loss, x_t, estimated_noise

    # def prior_vlb(self, x_0, args):
    #     """
    #     Get the prior KL term for the variational lower-bound, measured in
    #     bits-per-dim.
    #
    #     This term can't be optimized, as it only depends on the encoder.
    #
    #     :param x_0: the [N x C x ...] tensor of inputs.
    #     :param args: a dict of training arguments.
    #     :return: a batch of [N] KL values (in bits), one per batch element.
    #     """
    #     t = torch.tensor([self.num_timesteps - 1] * args['batch_size'], device=x_0.device)
    #     qt_mean, _, qt_log_variance = self.q_mean_variance(x_0, t)
    #     kl_prior = normal_kl(
    #         mean1=qt_mean, logvar1=qt_log_variance, mean2=torch.tensor(0.0, device=x_0.device),
    #         logvar2=torch.tensor(0.0, device=x_0.device)
    #     )
    #     return mean_flat(kl_prior) / np.log(2.0)

    # def calc_total_vlb(self, x_0, model, args):
    #     """
    #     Compute the entire variational lower-bound, measured in bits-per-dim,
    #     as well as other related quantities.
    #
    #     :param x_0: the [N x C x ...] tensor of inputs.
    #     :param model: the model to evaluate loss on.
    #     :param args: a dict of training arguments.
    #
    #     :return: a dict containing the following keys:
    #              - total_bpd: the total variational lower-bound, per batch element.
    #              - prior_bpd: the prior term in the lower-bound.
    #              - vb: an [N x T] tensor of terms in the lower-bound.
    #              - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
    #              - mse: an [N x T] tensor of epsilon MSEs for each timestep.
    #     """
    #     vb = []
    #     x_0_mse = []
    #     mse = []
    #     for t in reversed(list(range(self.num_timesteps))):
    #         t_batch = torch.tensor([t] * min(args['batch_size'], x_0.size()[0]), device=x_0.device)
    #         noise = self.noise_fn(x_0, t).float()
    #         x_t = self.q_sample(x_0=x_0, t=t_batch, noise=noise)
    #         # Calculate VLB term at the current timestep
    #         with torch.no_grad():
    #             out = self.calc_vlb_xt(
    #                 model,
    #                 x_0=x_0,
    #                 x_t=x_t,
    #                 t=t_batch,
    #             )
    #         vb.append(out["output"])
    #         x_0_mse.append(mean_flat((out["pred_x_0"] - x_0) ** 2))
    #         eps = self.predict_eps_from_x_0(x_t, t_batch, out["pred_x_0"])
    #         mse.append(mean_flat((eps - noise) ** 2))
    #
    #     vb = torch.stack(vb, dim=1)
    #     x_0_mse = torch.stack(x_0_mse, dim=1)
    #     mse = torch.stack(mse, dim=1)
    #
    #     prior_vlb = self.prior_vlb(x_0, args)
    #     total_vlb = vb.sum(dim=1) + prior_vlb
    #     return {
    #         "total_vlb": total_vlb,
    #         "prior_vlb": prior_vlb,
    #         "vb": vb,
    #         "x_0_mse": x_0_mse,
    #         "mse": mse,
    #     }
