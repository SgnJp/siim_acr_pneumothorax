import glob, os, time
import argparse
import tqdm
import pandas as pd
import numpy as np
import tarfile

import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dtorch.losses import dice_loss
from sklearn.model_selection import StratifiedKFold

import sys
import datetime

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, Normalize,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomGamma, CLAHE, RandomScale,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Resize, Rotate, ElasticTransform, RandomBrightness, RandomContrast)

from dtorch.dataset.dataset import SegmentationDataset
from dtorch.models.sedensenet import MultiSEDensenet121
from dtorch.utils.metrics import MetricsCallback, batch_dice_coeff, confusion_matrix_segmentation
from dtorch.utils.utils import test, train
from dtorch.utils.rle_tools import rle2area

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training_loop(input_dir : str, 
                  output_dir : str, 
                  img_size_x : int, 
                  img_size_y : int, 
                  batch_size : int,
                  num_epochs_1 : int, 
                  num_epochs_2 : int, 
                  lr_1 : float,
                  lr_2 : float,
                  gradient_accumulation : int,
                  cv_fold : int,
                  num_workers : int,
                  model_type : str,
                  model_fname : str = ""):


    ################# read data ######################
    df = pd.read_csv(os.path.join(input_dir, 'folds.csv'))
    df[" EncodedPixels"] = df["EncodedPixels"]
    df = df[df[' EncodedPixels'] != '-1']

    df['ImagePath'] = df['ImageId'].apply(lambda x : os.path.join(input_dir, 'train', x) + '.dcm')

    train_image_ids = set(df[df['fold'] != cv_fold]["ImageId"])
    train_msk = df["ImageId"].apply(lambda x : x in train_image_ids)

    train_df = df[train_msk]
    val_df = df[~train_msk]

    ################# prepare data loader #############
    tfms = Compose([
        Resize(img_size_x, img_size_y),
        HorizontalFlip(always_apply=False, p=0.5),
        ElasticTransform(border_mode=0, p=0.2),
        GridDistortion(border_mode=0, p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, interpolation=1, border_mode=0, always_apply=False, p=1.0),

        RandomBrightness(),
        GaussNoise(),
        Normalize(mean=(0.49, 0.49, 0.49), std=(0.235, 0.235, 0.235), max_pixel_value=255.0, always_apply=True, p=1.0)
    ], p=1.0)
    
    tfms_val = Compose([
        Resize(img_size_x, img_size_y),
        Normalize(mean=(0.49, 0.49, 0.49), std=(0.235, 0.235, 0.235), max_pixel_value=255.0, always_apply=True, p=1.0)
    ], p=1.0)


    train_ds = SegmentationDataset(train_df, tfms = tfms)
    val_ds = SegmentationDataset(val_df, tfms = tfms_val)

    print (len(train_ds), len(val_ds))
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)



    def combined_loss(output, target):
        return nn.BCEWithLogitsLoss()(output.view(-1,), target.view(-1,)) + dice_loss(output, target)
    #################### define metrics ###################
    metrics_callback = MetricsCallback()
    metrics_callback.add_callback("loss", lambda output, target : combined_loss(output, target).cpu().detach())
    metrics_callback.add_callback("dice, 0.2", lambda output, target : batch_dice_coeff(output.exp(), target, 0.2).cpu())
    metrics_callback.add_callback("dice, 0.3", lambda output, target : batch_dice_coeff(output.exp(), target, 0.3).cpu())
    metrics_callback.add_callback("dice, 0.4", lambda output, target : batch_dice_coeff(output.exp(), target, 0.4).cpu())
    metrics_callback.add_callback("dice, 0.5", lambda output, target : batch_dice_coeff(output.exp(), target, 0.5).cpu())
    metrics_callback.add_callback("dice, 0.6", lambda output, target : batch_dice_coeff(output.exp(), target, 0.6).cpu())
    metrics_callback.add_callback("dice, 0.7", lambda output, target : batch_dice_coeff(output.exp(), target, 0.7).cpu())
    metrics_callback.add_callback("dice, 0.8", lambda output, target : batch_dice_coeff(output.exp(), target, 0.8).cpu())
    metrics_callback.add_callback("dice, 0.9", lambda output, target : batch_dice_coeff(output.exp(), target, 0.9).cpu())



    model = smp.Unet(model_type, encoder_weights='imagenet', classes=1, activation='sigmoid')
    model = model.to(device)

    print ("Device count:", torch.cuda.device_count())


    def run(model, train_dl, val_dl, optimizer, loss, scheduler=None, metrics_callback=None, num_epochs=1, gradient_accumulation=1, prefix=""):

        for epoch in range(num_epochs):
            metrics_callback.reset()
            train_loss = train(model, train_dl, optimizer, loss, metrics_callback, None, gradient_accumulation=gradient_accumulation, device=device)
            print ("train: ", str(metrics_callback))
    
            metrics_callback.reset()
            val_loss = test(model, val_dl, metrics_callback, device=device)
            print ("valid: ", str(metrics_callback))
            if not scheduler is None:
                scheduler.step(max(metrics_callback.get()[2:]))
    
            torch.save(model, os.path.join(output_dir, prefix + str(epoch) + ".pth"))
            print ("Epoch ", epoch, " is done!")


    params = list(model.decoder.parameters())

    optimizer = optim.Adam(params, lr=lr_1)
    run(model, train_dl, val_dl, optimizer, combined_loss, None, metrics_callback, num_epochs_1, gradient_accumulation, "model_freeze_")
    
    optimizer = optim.Adam(model.parameters(), lr=lr_2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    run(model, train_dl, val_dl, optimizer, combined_loss, scheduler, metrics_callback, num_epochs_2, gradient_accumulation, "model_")
