import time
import os, sys
import tarfile
import json
from shutil import copyfile

from train import training_loop

HYPERPARAMETERS_PATH = "hyperparameters.json"

def strip_quotes(args):
    for k in args:
        args[k] = args[k].strip('\"')
    return args


if __name__ == "__main__":
    with open(HYPERPARAMETERS_PATH) as json_file:  
        args = json.load(json_file)

    args = strip_quotes(args)

    os.environ["CUDA_VISIBLE_DEVICES"]=args["gpu_id"]
    output_dir = args["output_dir"]
    input_dir = args["input_dir"]
 
    training_loop(input_dir             = input_dir,
                  output_dir            = output_dir, 
                  img_size_x            = int(args["img_size_x"]), 
                  img_size_y            = int(args["img_size_y"]), 
                  batch_size            = int(args["batch_size"]),
                  num_epochs_1          = int(args["num_epochs_1"]),
                  num_epochs_2          = int(args["num_epochs_2"]),
                  lr_1                  = float(args["lr_1"]),
                  lr_2                  = float(args["lr_2"]),
                  gradient_accumulation = int(args["gradient_accumulation"]),
                  cv_fold               = int(args["cv_fold"]),
                  num_workers           = 8,
                  model_type            = args["model_type"],
                  model_fname           = args["model_fname"])
