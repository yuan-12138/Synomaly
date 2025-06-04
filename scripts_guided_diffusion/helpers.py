"""Forked from https://github.com/Julian-Wyatt/AnoDDPM/blob/3052f0441a472af55d6e8b1028f5d3156f3d6ed3/helpers.py"""

import json
from collections import defaultdict

import torch
import torchvision.utils


def gridify_output(images, row_size=-1):
    scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    scaled_images = scale_img(images)

    # images = [(img - img.min()) / (img.max() - img.min()) * 255 for img in images]  # scale each image to [0, 255]
    # scaled_images = torch.stack(images).to(torch.uint8)

    return (torchvision.utils.make_grid(scaled_images, nrow=row_size, pad_value=-1).cpu().data.
            permute(0, 2, 1).contiguous().permute(2, 1, 0))


def defaultdict_from_json(json_dict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(json_dict)
    return dd


def get_args_from_json(json_file_name, server):
    """
    Load the arguments from a json file

    :param json_file_name: JSON file name
    :param server: server name
    :return: loaded arguments
    """

    if server == 'IFL':
        with open(f'/home/polyaxon-data/outputs1/lucie_huang/json_args/{json_file_name}.json', 'r') as f:
            args = json.load(f)
        args = defaultdict_from_json(args)

        args['json_file_name'] = json_file_name

        if args['dataset'].lower() == "brats23":
            args['data_path'] = "/home/polyaxon-data/data1/Lucie"
            args['output_path'] = "/home/polyaxon-data/outputs1/lucie_huang/gd_brats23"
        elif args['dataset'].lower() == "ultrasound":
            args['output_path'] = "/home/polyaxon-data/outputs1/lucie_huang/ultrasound"
        else:
            raise ValueError(f"Unsupported dataset {args['dataset']}")

    elif server == 'TranslaTUM':
        with open(f'/home/data/lucie_huang/json_args/{json_file_name}.json', 'r') as f:
            args = json.load(f)
        args = defaultdict_from_json(args)

        args['json_file_name'] = json_file_name

        if args['dataset'].lower() == "brats23":
            args['output_path'] = "/home/data/lucie_huang/brats23"
        elif args['dataset'].lower() == "ultrasound":
            args['output_path'] = "/home/data/lucie_huang/ultrasound"
        elif args['dataset'].lower() == "lits":
            args['output_path'] = "/home/data/lucie_huang/lits"
        else:
            raise ValueError(f"Unsupported dataset {args['dataset']}")

    else:
        with open(f'json_args/{json_file_name}.json', 'r') as f:
            args = json.load(f)
        args = defaultdict_from_json(args)

        args['json_file_name'] = json_file_name

        if args['dataset'].lower() == "brats23":
            args['data_path'] = "../data/BraTS2023"
            args['output_path'] = "../output/BraTS"
        elif args['dataset'].lower() == "ultrasound":
            args['output_path'] = "../output/Ultrasound"
        elif args['dataset'].lower() == "lits":
            args['output_path'] = "../output/LiTS"
        else:
            raise ValueError(f"Unsupported dataset {args['dataset']}")
    return args
