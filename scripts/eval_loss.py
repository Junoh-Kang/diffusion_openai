"""
Train a diffusion model on images.
"""

import argparse
import os
import torch
import numpy as np
import functools
from tqdm.auto import tqdm 

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()

    # Load Model
    path = '/data1/junoh/2022_DM/results/model/openai_celeba_full_channels='
    model_path = path + '{}/'.format(args.num_channels) + args.model_name
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    model.to(dist_util.dev())

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    losses = []
    t_list = list(range(4000))
    for t in range(4000):
        losses.append([])
    with torch.no_grad():
        for i in range(16):
            print('{}th loop'.format(i))
            batch, cond = next(data)
            batch = batch.to(dist_util.dev())

            for t in tqdm(range(4000)):
                ts = torch.tensor([t]*args.batch_size).to(dist_util.dev())
                compute_losses = functools.partial(
                    diffusion.training_losses,
                    model,
                    batch,
                    ts,
                    model_kwargs=cond,
                )   
                loss = compute_losses()
                #loss = diffusion.training_losses(model, batch, ts)
                losses[t].append(loss['loss'].cpu())
    for i in range(4000):
        losses[i] = torch.cat(losses[i])
    losses = torch.stack(losses)
    np.save('./{}'.format(args.num_channels), losses)
    breakpoint()

def create_argparser():
    defaults = dict(
        dataset_name="full",
        model_name="model180000.pt",
        data_dir="",
        schedule_sampler="uniform",
        image_size=128,
        num_res_blocks=3,
        learn_sigma=True,
        diffusion_steps=4000,
        noise_schedule='cosine',
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=16,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        # log_interval=10,
        # save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
