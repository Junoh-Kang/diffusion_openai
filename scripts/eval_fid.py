import numpy as np
import torch

from pytorch_gan_metrics import get_fid
import imageio
import glob
"""
https://github.com/w86763777/pytorch-gan-metrics
python -m pytorch_gan_metrics.calc_fid_stats --path path/to/images --output name.npz
Prepare images in type torch.float32 with shape [N, 3, H, W] and normalized to [0,1]
"""

def load_images(image_path):
    f_list = list(glob.glob(image_path + '*.png'))
    images = []
    for f in f_list:
        images.append(imageio.imread(f))
    images = np.stack(images)/255
    images = torch.tensor(images, dtype=torch.float32).permute(0,3,1,2)
    return images


image_path = '/data1/junoh/2022_DM/results/sample/openai_celeba_full/'
# image_path = image_path + 'ddpm_c128model180000_to_c32model180000/'
# f_list = list(glob.glob(image_path + 'model*'))

image_path = image_path + 'ddpm_c32model180000_to_c128model180000/'
f_list = list(glob.glob(image_path + 'model*'))

# image_path = image_path + 'ddpm_c32model180000/'
# f_list = list(glob.glob(image_path + 'nor*'))


images_set = {}
for f in f_list:
    name = str.split(f, '/')[-1]
    print(name)
    images = load_images(f + '/sample/')
    images_set[name] = images

data_path = '/data1/junoh/2022_DM/data/celeba_full.npz'

with open(image_path + 'FID.txt', 'w') as text_file:
    for key in sorted(images_set.keys()):
        fid = get_fid(images_set[key], data_path)
        text_file.write(key + ' FID: {}'.format(fid) + '\n')
