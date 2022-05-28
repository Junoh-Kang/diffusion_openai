"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from torchvision import utils
import imageio
from tqdm.auto import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
"""
This script is to sample from the scratch. 
It will save both sampling process and Final sample
It will sample through two Models 

python script/image_sample_process_mixed.py 
    --num_channels : first model channel
    --num_channels2 : second model channel
    --model_change_at : when to change to num_channels, 
                        -1 : use only model 1
                        3999 : use only model 2
"""
def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(True)

    # Load Model
    path = '/data1/junoh/2022_DM/results/model/openai_celeba_full_channels='
    # Load Model 1
    model_path1 = path + '{}/'.format(args.num_channels) + args.model_name
    model1, diffusion1 = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model1.load_state_dict(
        dist_util.load_state_dict(model_path1, map_location="cpu")
    )
    model1.to(dist_util.dev())
    if args.use_fp16:
        model1.convert_to_fp16()
    model1.eval()
    
    # Load Model 2 
   
    args.num_channels, args.num_channels2 = args.num_channels2, args.num_channels

    model_path2 = path + '{}/'.format(args.num_channels) + args.model_name2
    model2, diffusion2 = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )   
    model2.load_state_dict(
        dist_util.load_state_dict(model_path2, map_location="cpu")
    )   
    model2.to(dist_util.dev())
    if args.use_fp16:
        model2.convert_to_fp16()
    model2.eval()

    args.num_channels, args.num_channels2 = args.num_channels2, args.num_channels
    
    # Set save Path
    image_path = '/data1/junoh/2022_DM/results/sample/openai_celeba_' + args.dataset_name + '/'
    model_name = 'c{}'.format(args.num_channels) + str.split(args.model_name, '.')[0] \
               + '_to_c{}'.format(args.num_channels2) + str.split(args.model_name2, '.')[0]
    if args.use_ddim:
        image_path = image_path + 'ddim_' + model_name + '/' 
    else:
        image_path = image_path + 'ddpm_' + model_name + '/'
    
    image_path = image_path + 'model_change_at{}/'.format(args.model_change_at)
    os.makedirs(image_path + 'sample_process', exist_ok=True)
    os.makedirs(image_path + 'sample', exist_ok=True)
    
    for image_num in range(args.num_start, args.num_start + args.num_samples):
        model_kwargs = {}

        if args.use_ddim:
            sample_fn1 = diffusion1.ddim_sample
            sample_fn2 = diffusion2.ddim_sample
        else:
            sample_fn1 = diffusion1.p_sample
            sample_fn2 = diffusion2.p_sample

        sample_fn = sample_fn1
        model = model1
        
        history = {'sample':[], 'pred_xstart':[]}
        model_kwargs = {}

        th.manual_seed(image_num)
        img = th.randn(1,3,128,128, device='cuda')

        indices = tqdm(list(range(args.diffusion_steps))[::-1])
        for i in indices:
            t = th.tensor([i]*1, device='cuda')
            # change model
            if t == args.model_change_at:
                print("changed") 
                sample_fn = sample_fn2
                model = model2

            with th.no_grad():
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                img = out['sample']

            show_t = list(range(3999, 3900, -10)) + [0]
            if t in show_t:
                history['sample'].append(out['sample'])
                history['pred_xstart'].append(out['pred_xstart'])

        sample = th.cat([*history['sample'], *history['pred_xstart']])
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample_process = utils.make_grid(sample, nrow=11).permute(1,2,0).cpu().numpy()
        sample_image = sample[-1].permute(1,2,0).cpu().numpy()

        imageio.imwrite(image_path + 'sample_process/{}.png'.format(image_num), sample_process)
        imageio.imwrite(image_path + 'sample/{}.png'.format(image_num), sample_image)

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_start=0,
        num_samples=16,
        batch_size=1,
        use_ddim=False,
        dataset_name="full",
        num_channels2=32,
        model_name="model180000.pt",
        model_name2="model180000.pt",
        model_change_at=-1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
