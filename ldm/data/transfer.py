import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TransferBase(Dataset):
    def __init__(self,
                 txt_file,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths = txt_file
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        pathA = self.image_paths[i].split()[0]
        pathB = self.image_paths[i].split()[1]
        imageA = Image.open(pathA)
        imageB = Image.open(pathB)

        # default to score-sde preprocessing
        imgA = np.array(imageA).astype(np.uint8)
        imgB = np.array(imageB).astype(np.uint8)
        
        crop = min(imgA.shape[0], imgA.shape[1])
        h, w, = imgA.shape[0], imgA.shape[1]
        imgA = imgA[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        crop = min(imgB.shape[0], imgB.shape[1])
        h, w, = imgB.shape[0], imgB.shape[1]
        imgB = imgB[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        
        imageA = Image.fromarray(imgA)
        imageB = Image.fromarray(imgB)
        if self.size is not None:
            imageA = imageA.resize((self.size, self.size), resample=self.interpolation)
            imageB = imageB.resize((self.size, self.size), resample=self.interpolation)

        imageA = self.flip(imageA)
        imageA = np.array(imageA).astype(np.uint8)
        imageB = np.array(imageB).astype(np.uint8)

        imageA = (imageA / 127.5 - 1.0).astype(np.float32)
        imageB = (imageB / 127.5 - 1.0).astype(np.float32)
        example["imageA"] = np.stack([imageA, imageA, imageA], axis=2)
        example["imageB"] = np.stack([imageB, imageB, imageB], axis=2)
        return example


class TransferTrain(TransferBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="CT-MR/train_list.txt", **kwargs)


class TransferValidation(TransferBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="CT-MR/val_list.txt", flip_p=flip_p, **kwargs)

# a = TransferTrain()

# for x in a:
#     print(x['imageA'].shape, x['imageB'].shape) 