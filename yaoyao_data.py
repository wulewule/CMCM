import os, random
import numpy as np
import PIL, cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A

class TransferBase(Dataset):
    def __init__(self, txt_file1, txt_file2):
        self.data_paths1 = txt_file1
        self.data_paths2 = txt_file2
        
        with open(self.data_paths1, "r") as f1:
            self.image_paths1 = f1.read().splitlines()
            
        with open(self.data_paths2, "r") as f2:
            self.image_paths2 = f2.read().splitlines()    

        self.image_paths = self.image_paths1 + self.image_paths2
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
        
        item_A = cv2.resize(item_A, (256,256), cv2.INTER_AREA)
        item_B = cv2.resize(item_B, (256,256), cv2.INTER_AREA)
        
        image_A = np.stack([item_A, item_A, item_A], axis=0)
        image_B = np.stack([item_B, item_B, item_B], axis=0)
        
        example["imageA"] = torch.tensor(image_A)
        example["imageB"] = torch.tensor(image_B)
        example["class_label"] = int(label)
        return example


class TransferTrain(TransferBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file1="PET-CT/train_list.txt", txt_file2="PET-MR/train_list.txt")


class TransferValidation(TransferBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file1="PET-CT/valid_list.txt", txt_file2="PET-MR/valid_list.txt")
