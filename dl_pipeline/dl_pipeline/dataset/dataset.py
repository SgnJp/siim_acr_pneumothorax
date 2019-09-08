import os
import zipfile, tarfile
import pandas as pd
import glob
import numpy as np
import cv2
import scipy.io as sio
import logging
import errno
import pydicom

from torchvision import transforms
from torch.utils.data import Dataset
import random

from dtorch.utils.rle_tools import rle2mask

from torch.utils.data import Dataset
from torchvision import transforms


class ImageClassificationDataset(Dataset):

    def __init__(self,
                 input_path,
                 img_ids,
                 labels,
                 tfms = None):

        self.input_path = input_path
        self.img_ids = img_ids
        self.labels = labels
        self.tfms = tfms

		## TODO: sanity check, types of tfms labels
		
        assert len(self.img_ids) == len(self.labels)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
		## TODO: sanity check, file exists
        img_path = os.path.join(self.input_path, self.img_ids[idx])

        img = cv2.imread(img_path)
        
        data = {"image": img}
        augmented = self.tfms(**data)
        img = augmented["image"]

        return transforms.ToTensor()(img), torch.from_numpy(self.labels[idx])
		
		
class SegmentationDataset(Dataset):

    def __init__(self,
                 df,
                 tfms = None):

        self.df = df
        self.ids_list = self.df["ImageId"].unique()

        self.tfms = tfms

    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, idx):
        img_id = self.ids_list[idx]

        df_masks = self.df[self.df["ImageId"] == img_id]
        img_path = df_masks.iloc[0]["ImagePath"]

        img = pydicom.read_file(img_path).pixel_array
        segm = np.zeros_like(img)

        img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)

        if df_masks.iloc[0][' EncodedPixels'] != ' -1':
            for i in range(len(df_masks)):
                segm += rle2mask(df_masks.iloc[i][' EncodedPixels'], img.shape[0], img.shape[1]).astype(np.uint8).T

        segm = np.expand_dims(segm, axis=2)

        data = {"image": img, "mask": segm}

        augmented = self.tfms(**data)
        img, segm = augmented["image"], augmented["mask"]

        return transforms.ToTensor()(img), transforms.ToTensor()(segm)

class ExternalSegmentationDataset(Dataset):

    def __init__(self,
                 names,
                 segm,
                 input_dir,
                 tfms = None):

        self.names = names
        self.segm = segm

        self.tfms = tfms
        self.input_dir = input_dir

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.input_dir, self.names[idx])
    
        img = cv2.imread(img_path)

        data = {"image": img, "mask" : self.segm[idx]}

        augmented = self.tfms(**data)
        img, segm = augmented["image"], augmented["mask"]
        segm = np.expand_dims(segm, axis=2)

        return transforms.ToTensor()(img), transforms.ToTensor()(segm)

class ExternalClassificationDataset(Dataset):

    def __init__(self,
                 df,
                 tfms = None):

        self.df = df

        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
    
        img_path = item["ImagePath"]
        img = cv2.imread(img_path)

        data = {"image": img}

        augmented = self.tfms(**data)
        img= augmented["image"]

        return transforms.ToTensor()(img), item['Pneumothorax']  * 1.0

class ClassificationDataset(Dataset):

    def __init__(self,
                 df,
                 tfms = None):

        self.df = df
        self.ids_list = self.df["ImageId"].unique()

        self.tfms = tfms

    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, idx):
        img_id = self.ids_list[idx]

        df_masks = self.df[self.df["ImageId"] == img_id]
        img_path = df_masks.iloc[0]["ImagePath"]

        img = pydicom.read_file(img_path).pixel_array
        segm = np.zeros_like(img)

        img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)

        data = {"image": img}
        augmented = self.tfms(**data)
        img = augmented["image"]

        return transforms.ToTensor()(img), (df_masks.iloc[0][' EncodedPixels'] != ' -1')  * 1.0
