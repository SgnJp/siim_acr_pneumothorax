# siim_acr_pneumothorax

# SIIM-ACR Pneumothorax Segmentation (10-th place)

The pipeline to do training and inference on segmentation data.

### Installing

All you need before is that pytorch with the gpu support is installed.
Afterwards you can clone the repository and install the package using pip.

```
pip install -e dl_pipeline
```

## Solution
To read more about the kaggle competition, please read:
https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation

The solution itself is described in the topic:
https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107687

The whole pipeline consist of segmentation and classification models.

## Training of classification
For this all you need is the package itself: dl\_pipeline and everything from the classification folder:
script that runs the training and hyperparameters.

Example how to train the first fold of se\_resnext101 classifier:
```
python classification/run.py classification/hyperparameters/hyperparameters_res101_0.json
```

Example how to train the first fold of senet154 classifier:
```
python classification/run.py classification/hyperparameters/hyperparameters_senet154_0.json
```

After running all training scripts for all folds, you will have all necessery classification models.

## Training of segmentation
For this all you need is the package itself: dl\_pipeline and everything from the segmentation folder:
script that runs the training and hyperparameters.

Example how to train the first fold of dpn98 segmetation:
```
python segmentation/run.py segmentation/hyperparameters/hyperparameters_dpn98_512_0.json
python segmentation/run.py segmentation/hyperparameters/hyperparameters_dpn98_1024_0.json
```

Then you have to repeat this procedure 15 times: for each of three models dense121, dpn98 and resnet101 and 
each of 5 folds.

Afterwards you will be able to use this models to do th inference.

## Inference

To do the inference please run the notebook submission/prepare\_submission.ipynb with the 
specified models, that are trained in the previous step
