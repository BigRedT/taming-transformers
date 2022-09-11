import io
import os
import PIL
from PIL import Image
import torch
import requests
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))


def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def preprocess(img, target_image_size=256, map_dalle=True):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    # img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    if map_dalle: 
      img = map_pixels(img)
    return img

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def load_model(name):
    if name=='coco':
        config_path = 'logs/vqgan_coco/configs/model.yaml'
        config = OmegaConf.load(config_path).model.params.first_stage_config.params
        ckpt_path = '/home/tanmayg/Data/taming/checkpoints/vqgan_coco.ckpt'
    elif name=='imagenet':
        config_path = 'logs/vqgan_imagenet_f16_16384/configs/model.yaml'
        config = OmegaConf.load(config_path).model.params
        ckpt_path = 'logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt'
    
    model = VQModel(**config)
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = 'imagenet'

url = 'http://farm9.staticflickr.com/8361/8386661993_3a0e803ec8_z.jpg'
# url = 'https://heibox.uni-heidelberg.de/f/7bb608381aae4539ba7a/?dl=1'
img = download_image(url)
# img = Image.open('assets/elephant.png').convert('RGB')
x = preprocess(img,target_image_size=384,map_dalle=False).to(DEVICE)
x = preprocess_vqgan(x)
preproc_img = custom_to_pil(x[0])

model = load_model(model_name).to(DEVICE)
z, _, [_, _, indices] = model.encode(x)
# z_small=torch.nn.functional.interpolate(z,scale_factor=(0.5,0.5))
# z_large = torch.nn.functional.interpolate(z_small,scale_factor=(2,2))
z_modified = z
z_modified[:,:,12:,:] = 0
# z_modified[:,:,1::2,1::2]=0
x_hat = model.decode(z_modified)[0]
recon_img = custom_to_pil(x_hat)

outdir = f'logs/recon/{model_name}'
os.makedirs(outdir,exist_ok=True)
img.save(os.path.join(outdir,'original.jpg'))
preproc_img.save(os.path.join(outdir,'preproc.jpg'))
recon_img.save(os.path.join(outdir,'recon.jpg'))

import pdb; pdb.set_trace()