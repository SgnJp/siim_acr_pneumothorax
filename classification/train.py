import glob, os, time
import argparse
import tqdm
import pandas as pd
import numpy as np
import tarfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold

import sys
import datetime

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, Normalize,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomGamma, CLAHE, RandomScale,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Resize, Rotate, ElasticTransform, RandomBrightness, RandomContrast)

from dl_pipeline.dataset.dataset import SegmentationDataset, ClassificationDataset, ExternalClassificationDataset
from dl_pipeline.dataset.auxiliary import AggregatorDataset
#from dl_pipeline.models.unets import Unet34
from dl_pipeline.models.classification import SEResNetx101, SENet154
from dl_pipeline.utils.metrics import MetricsCallback, batch_dice_coeff, confusion_matrix_segmentation, accuracy
from dl_pipeline.utils.utils import test, train, predict
from dl_pipeline.utils.rle_tools import rle2area

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PseudoLabeling:
    def __init__(self, input_dir, tfms_val):
        ext_df = pd.read_csv(os.path.join(input_dir, 'external/data/Data_Entry_2017.csv'))
        self.unlabeled_images = ext_df["Image Index"].apply(lambda x : os.path.join(input_dir, "external/data/images", x)).values
        self.unlabeled_images = self.unlabeled_images
        self.labeled_images = []
        self.labels = []

        self.tfms_val = tfms_val

    def update(self, model):
        ext_df = pd.DataFrame(np.array([self.unlabeled_images, [1.0]*len(self.unlabeled_images)]).T, columns=["ImagePath", "Pneumothorax"])

        ext_ds = ExternalClassificationDataset(ext_df, tfms = self.tfms_val)
        ext_dl = torch.utils.data.DataLoader(ext_ds, batch_size=20, shuffle=False, num_workers=8)
        
        n_pos_inst = 300
        n_neg_inst = 5000

        predictions = predict(model, ext_dl)
        predictions = np.squeeze(predictions)

        n_neg_idx = predictions.argsort()[:n_neg_inst]
        n_pos_idx = predictions.argsort()[-n_pos_inst:]

        
        self.labeled_images.extend(self.unlabeled_images[n_pos_idx])
        self.labeled_images.extend(self.unlabeled_images[n_neg_idx])

        self.unlabeled_images = np.delete(self.unlabeled_images, np.concatenate([n_pos_idx, n_neg_idx]))
        print (len(self.unlabeled_images))

        self.labels.extend([1.0]*n_pos_inst)
        self.labels.extend([0.0]*n_neg_inst)

    def get_ds(self):
        ext_df = pd.DataFrame(np.array([self.labeled_images, np.array(self.labels)]).T, columns=["ImagePath", "Pneumothorax"])
        return ExternalClassificationDataset(ext_df, tfms = self.tfms_val)
        


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
                  model_fname : str = ""):


    ################# read data ######################
    df = pd.read_csv(os.path.join(input_dir, 'folds.csv'))
    df[" EncodedPixels"] = df["EncodedPixels"]

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

    train_ds = ClassificationDataset(train_df, tfms = tfms)
    val_ds = ClassificationDataset(val_df, tfms = tfms_val)

    
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)



    def combined_loss(output, target):
        return nn.BCEWithLogitsLoss()(output.view(-1,), target.view(-1,)) #+ 0.3*F_loss(output.view(-1,), target.view(-1,), beta=0.5)

    def F_score(logit, label, threshold=0.5, beta=2):
        #prob = torch.sigmoid(logit)
        prob = logit > threshold
        label = label > 0.5

        TP = (prob & label).sum().float()
        TN = ((~prob) & (~label)).sum().float()
        FP = (prob & (~label)).sum().float()
        FN = ((~prob) & label).sum().float()

        precision = TP / (TP + FP + 1e-12)
        recall = TP / (TP + FN + 1e-12)
        F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
        return F2.mean(0)

    #################### define metrics ###################
    metrics_callback = MetricsCallback()
    metrics_callback.add_callback("loss", lambda output, target : combined_loss(output, target).cpu().detach())
    metrics_callback.add_callback("acc, 0.5",  lambda output, target : accuracy(output.view(-1,).cpu(), target.view(-1,).cpu()).cpu().detach())
    metrics_callback.add_callback("acc, 0.0",  lambda output, target : accuracy(output.view(-1,).cpu(), target.view(-1,).cpu(), 0.0).cpu().detach())
#    metrics_callback.add_callback("fscore, 0.8",  lambda output, target : F_score(output.view(-1,).cpu(), target.view(-1,).cpu(), 0.8, beta=0.5).cpu().detach())
#    metrics_callback.add_callback("fscore, 1.0",  lambda output, target : F_score(output.view(-1,).cpu(), target.view(-1,).cpu(), 1.0, beta=0.5).cpu().detach())
#    metrics_callback.add_callback("fscore, 1.2",  lambda output, target : F_score(output.view(-1,).cpu(), target.view(-1,).cpu(), 1.2, beta=0.5).cpu().detach())


    if model_type == "senet154":
        model = SEResNetx101(num_classes=1)
    elif model_type == "se_resnext101":
        model = SENet154(num_classes=1)


    if model_fname != "":
        model = torch.load(os.path.join("models", model_fname))

    model = model.to(device)
    print ("Device count:", torch.cuda.device_count())


    def run(model, train_dl, val_dl, optimizer, loss, scheduler=None, metrics_callback=None, num_epochs=1, gradient_accumulation=1, prefix=""):

        best_model = 0
        best_loss = 10000.0

        for epoch in range(num_epochs):
            metrics_callback.reset()
            train_loss = train(model, train_dl, optimizer, loss, metrics_callback, None, gradient_accumulation=gradient_accumulation, device=device)
            print ("train: ", str(metrics_callback))
    
            metrics_callback.reset()
            test(model, val_dl, metrics_callback, device=device)


            val_predictions = np.squeeze(predict(model, val_dl))
            val_targets = np.squeeze(np.concatenate(np.array([target.numpy() for _, target in val_dl])))
       
            print ("valid: ", str(metrics_callback))

            f_score_res = []
            for i in np.arange(-20, 30)*0.1:
                f_score_res.append((i, F_score(torch.tensor(val_predictions), torch.tensor(val_targets), threshold=i, beta=0.5).numpy()))
            print ("F0.8, F1.0, F1.2:", f_score_res[28], f_score_res[30], f_score_res[32])
            print ("FOpt: ", sorted(f_score_res, key = lambda x : x[1])[-1])

            if not scheduler is None:
                scheduler.step(max(metrics_callback.get()[2:]))


            val_loss = metrics_callback.get_by_name("loss")
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = epoch
                print ("New best loss:", best_loss)
 
            torch.save(model, os.path.join(output_dir, prefix + str(epoch) + ".pth"))
            print ("Epoch ", epoch, " is done!")

        return torch.load(os.path.join(output_dir, prefix + str(best_model) + ".pth"))

    ps = PseudoLabeling(input_dir, tfms_val)
    agg_ds = AggregatorDataset([leak_ds, train_ds])

    for k in range(10):
        model = torch.load(os.path.join("models", model_fname))
        optimizer = optim.Adam(model.parameters(), lr=lr_2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

        train_dl = torch.utils.data.DataLoader(agg_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        model = run(model, train_dl, val_dl, optimizer, combined_loss, scheduler, metrics_callback, 10, gradient_accumulation, "model_" + str(k) + "_")

        ps.update(model)
        agg_ds = AggregatorDataset([ps.get_ds(), leak_ds, train_ds])
        print ("Len of training:", len(agg_ds))
