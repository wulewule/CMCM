import os, PIL
from PIL import Image
import cv2
import math
import glob, random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch, sys
from torch import optim
from ldm.util import instantiate_from_config
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from functools import partial
from inspect import isfunction
from torch.optim import lr_scheduler
from torchvision.utils import save_image

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU=============")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


from ldm.modules.diffusionmodules.openaimodel import UNetModel

class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim=256, n_classes=2):
        super().__init__()
        
        self.embedding = nn.Embedding(n_classes, embed_dim)
    
    def forward(self, x):
        c = x[:, None]
        c = self.embedding(c)
        return c

class Encoder(nn.Module):
    def __init__(self, encodeconfig, ckpt_path=None):
        super().__init__()
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        self.instantiate_encode(encodeconfig)
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def disabled_train(self, mode=True):
        return self
    
    def instantiate_encode(self, config):
        self.first_stage_model = instantiate_from_config(config)
        self.first_stage_model.eval()
        self.first_stage_model.train = self.disabled_train
        for name, param in self.first_stage_model.named_parameters():  
            param.requires_grad = False

    @torch.no_grad()
    def encode_first(self, x):
        posterior = self.first_stage_model.encode(x)
        return posterior
    
    @torch.no_grad()
    def decode_first(self, z):
        rec = self.first_stage_model.decode(z)
        return rec

class Transfer(Dataset):
    def __init__(self,
                 txt_file1,
                 txt_file2,
                 data_transform=None,
                 ):
        self.data_paths1 = txt_file1
        self.data_paths2 = txt_file2
        self.data_transform = data_transform
        
        with open(self.data_paths1, "r") as f1:
            self.image_paths1 = f1.read().splitlines()
            
        with open(self.data_paths2, "r") as f2:
            self.image_paths2 = f2.read().splitlines()
        
        self.image_paths = []
        self.middle_paths = self.image_paths1+self.image_paths2
        
        for f in self.middle_paths:
            name = int(f.split('.')[-2].split('_')[-1])
            if name<90:
                self.image_paths.append(f)
        
        random.shuffle(self.image_paths)
        self._length = len(self.image_paths)
    
    def min_max(self, x, min_x, max_x):    
        y = (x-min_x)/(max_x-min_x)
        return y   

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        paths = self.image_paths[i].split()[0]
        label = self.image_paths[i].split()[1]
            
        imgs = np.load(paths)
        item_A = imgs['a']
        item_B = imgs['b']          
        #name = int(paths.split('/')[-1].split('_')[0])
        
        if label == '0':
            item_A[item_A>20000]=20000
            item_A = self.min_max(item_A, 0, 20000) 
        
            item_B[item_B>2000]=2000
            item_B = self.min_max(item_B, -1024, 2000) 
        
        else:
            item_A[item_A>20000]=20000
            item_A = self.min_max(item_A, 0, 20000) 
        
            item_B[item_B>10000]=10000
            item_B = self.min_max(item_B, 0, 10000) 
        
        item_B = cv2.resize(item_B, (256,256), cv2.INTER_AREA)
        item_A = cv2.resize(item_A, (256,256), cv2.INTER_AREA)
        
        image_A = np.stack([item_A, item_A, item_A], axis=0)
        image_B = np.stack([item_B, item_B, item_B], axis=0)
        return torch.tensor(image_A), torch.tensor(image_B), int(label)

def loadData(batch_size, prefix, shuffle=True):  
    if prefix == 'train':
        dataset = Transfer(txt_file1="PET-CT/train_list.txt", txt_file2="PET-MR/train_list.txt")
    else:
        dataset = Transfer(txt_file1="PET-CT/valid_list.txt", txt_file2="PET-MR/valid_list.txt")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def open_img(path):
    image = Image.open(path).convert('L')       
    image = np.array(image).astype(np.uint8)
    #image = Image.fromarray(image)
    #image = image.resize((256, 256), resample=PIL.Image.BICUBIC)
    #image = np.array(image).astype(np.uint8)
    return image

def d_min_max(x, min_x=-1024, max_x=31743):
    y = x * (max_x-min_x) + min_x
    return y  

def exists(x):
    return x is not None

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

def q_sample(x_start, t, noise=None):
    
    betas = make_beta_schedule("linear", 1000, linear_start=0.0015, linear_end=0.0195,
                                       cosine_s=8e-3)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
        
    to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
    sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod))
    sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod))
    noise = default(noise, lambda: torch.randn_like(x_start))
    return (extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


def main():
    save_path = 'output/'
    batch_size = 1
    image_size = 256
    train_loader = loadData(batch_size, 'train', shuffle=True)
    valid_loader = loadData(batch_size, 'valid', shuffle=False)
    print(len(train_loader), len(valid_loader))
    
    in_ch, out_ch = 3, 3  

    lr_G   = 5e-5
    epochs = 200
    num_timesteps = 1000
    
    config = OmegaConf.load("yaoyao.yaml")
    
    first_model = Encoder(config['model']['params']['encodeconfig']).to(device)
    
    unet_model  = UNetModel(
        image_size=64,
        in_channels=4,
        model_channels=256,
        out_channels=4,
        num_res_blocks=1,
        attention_resolutions=(4,2,1),
        dropout=0.5,
        channel_mult=(1,2,4),
        num_head_channels=16,
        use_spatial_transformer=True,
        context_dim=256).to(device)
    
    class_model = ClassEmbedder().to(device)
    #unet_model.load_state_dict(torch.load("cailast_dfmodel.ckpt"))
    #class_model.load_state_dict(torch.load("cailast_dfmodel.ckpt"))
    
    optimizer_G = optim.AdamW(list(unet_model.parameters())+list(class_model.parameters()) , lr=lr_G, weight_decay=0.01)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=int(3500/64*200), eta_min=1e-6)
    
    with torch.no_grad():
        unet_model.eval()
        class_model.eval()
        X, Y, C = next(iter(valid_loader))
        x = first_model.encode_first(X.to(device, dtype=torch.float)).mode()
        y = first_model.decode_first(first_model.encode_first(Y.to(device, dtype=torch.float)).mode())
        t = torch.randint(0, num_timesteps, (x.shape[0],), device=device).long()
        c = class_model(C.to(device))
        
        #noise =None
        #noise = default(noise, lambda: torch.randn_like(x)).to(device)
        #x_noisy = q_sample(x_start=x, t=t, noise=noise)
        g = first_model.decode_first(unet_model(x, t, c))
        print(X.shape, Y.shape, g.shape)
        save_image(X, save_path + 'input.png')
        save_image(y, save_path + 'ground-truth.png')
        save_image(g, save_path + 'sample_0.png')

    G_Losses, max_ssim = [], 0  
    gt_img = open_img('output/'+'ground-truth.png')  
    for epoch in range(epochs):
        G_losses, batch, g_l = [], 0, 0
        unet_model.train()
        class_model.train()
        
        dataset_size = 0
        running_loss = 0.0
        
        print('=======epoch--'+str(epoch))
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train')
        for step, (X, Y, C) in train_pbar:
            unet_model.zero_grad()
            class_model.zero_grad()  
            
            x = first_model.encode_first(X.to(device, dtype=torch.float)).sample()
            y = first_model.encode_first(Y.to(device, dtype=torch.float)).sample()
            t = torch.randint(0, num_timesteps, (x.shape[0],), device=device).long()
            c = class_model(C.to(device))
            
            noise =None
            noise = default(noise, lambda: torch.randn_like(x)).to(device)
            x_noisy = q_sample(x_start=x, t=t, noise=noise)
            columns_to_zero = torch.randperm(x_noisy.shape[0])[:x_noisy.shape[0]//4]
            x_noisy[columns_to_zero,:,:,:] = 0
           
            g = unet_model(x+x_noisy, t, c)

            G_loss = (nn.MSELoss()(g, y.to(device, dtype=torch.float)))
            G_loss.backward()
            optimizer_G.step()
            scheduler.step()
            
            running_loss += (G_loss.item() * batch_size)
            dataset_size += batch_size
        
            epoch_loss = running_loss / dataset_size
            current_lr = optimizer_G.param_groups[0]['lr']
            train_pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}', lr=f'{current_lr:0.6f}')
        
        torch.cuda.empty_cache()
        G_Losses.append(epoch_loss)
           
        with torch.no_grad():
            unet_model.eval()
            class_model.eval()
            
            dataset_size = 0
            running_loss = 0.0
            
            valid_pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid')
            for step, (X, Y, C) in valid_pbar:
                x = first_model.encode_first(X.to(device, dtype=torch.float)).sample()
                y = first_model.encode_first(Y.to(device, dtype=torch.float)).sample()
                t = torch.randint(0, num_timesteps, (x.shape[0],), device=device).long()
                c = class_model(C.to(device))

                g = unet_model(x, t, c)
    
                G_loss = (nn.MSELoss()(g, y.to(device, dtype=torch.float)))
                
                running_loss += (G_loss.item() * batch_size)
                dataset_size += batch_size
            
                epoch_loss = running_loss / dataset_size
                valid_pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}')
            
            if (epoch+1)%10 == 0:
                X, Y, C = next(iter(valid_loader))
                x = first_model.encode_first(X.to(device, dtype=torch.float)).sample()
                t = torch.randint(0, num_timesteps, (x.shape[0],), device=device).long()
                c = class_model(C.to(device))

                g = first_model.decode_first(unet_model(x, t, c))
                save_image(g, save_path + 'sample_' + str(epoch + 1) + '.png')
                
                X, Y, C = next(iter(train_loader))
                x = first_model.encode_first(X.to(device, dtype=torch.float)).sample()
                t = torch.randint(0, num_timesteps, (x.shape[0],), device=device).long()
                c = class_model(C.to(device))

                g = first_model.decode_first(unet_model(x, t, c))
                save_image(g, save_path + 'train_sample' + '.png')
                save_image(Y, save_path + 'train_truth' + '.png')
                
        
        torch.save(unet_model.state_dict(), 'ldm_model.ckpt')
        torch.save(class_model.state_dict(), 'ldm_classmodel.ckpt')
        torch.cuda.empty_cache()
    

def min_max(x, min_x, max_x):    
    y = (x-min_x)/(max_x-min_x)
    return y 

def d_min_max(x, min_x, max_x):
    y = x * (max_x-min_x) + min_x
    # print(np.unique(im), type(im))
    return y
        
if __name__ == '__main__':
    main()
