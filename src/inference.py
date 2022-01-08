import json
import os
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage import io, img_as_uint
from tqdm import tqdm_notebook, tqdm
from zipfile import ZipFile
from predict import Model
from predict import load_data
from utils import imsetshow
import torch
from DataLoader import ImagesetDataset, ImageSet
from DeepNetworks.HRNet import HRNet
from Evaluator import shift_cPSNR
from utils import getImageSetDirectories, readBaselineCPSNR, collateFunction


def load_model(config, checkpoint_file):
    '''
    Loads a pretrained model from disk.
    Args:
        config: dict, configuration file
        checkpoint_file: str, checkpoint filename
    Returns:
        model: HRNet, a pytorch model
    '''

    #   checkpoint_dir = config["paths"]["checkpoint_dir"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HRNet(config["network"]).to(device)
    temp = torch.load(checkpoint_file)
    model.load_state_dict(temp)
    return model

# model_file ='../models/weights/batch_8_views_32_min_32_beta_50.0_time_2021-12-10-20-32-43-747510'
# config_add = '../config/config.json'
# with open(config_add, "r") as read_file:
#     config = json.load(read_file)
# model = load_model(config,model_file)
# print(model.config(config,))
#
config_file_path = "../config/config.json"
with open(config_file_path, "r") as read_file:
    config = json.load(read_file)
print(config)

checkpoint_dir = config["paths"]["checkpoint_dir"]
run_subfolder = 'batch_8_views_32_min_32_beta_50.0_time_2021-12-10-20-32-43-747510'
checkpoint_filename = 'HRNet.pth'
checkpoint_file = os.path.join('..', checkpoint_dir, run_subfolder, checkpoint_filename)
# print(checkpoint_file)
print(checkpoint_file)
assert os.path.isfile(checkpoint_file)
model = Model(config)
model.load_checkpoint(checkpoint_file=checkpoint_file)
is_one_img = False
if(is_one_img):
    imset = ImagesetDataset(imset_dir="../data/test/NIR/imgset1400", config=config["training"], top_k=-1)
else :
    train_dataset, val_dataset, test_dataset, baseline_cpsnrs = load_data(config_file_path, val_proportion=0.10, top_k=-1)

    print("dataset size : ", len(train_dataset))
    results = model.evaluate(train_dataset[:5], val_dataset[:5], test_dataset[:10], baseline_cpsnrs)
    print(results.describe())
    print(results.loc[results['part']=='test'].describe())
    print(results.loc[results['part']=='test'].describe().loc['mean'])
    img_name = val_dataset[1]['name']
    imset = val_dataset[img_name]

print(len(imset))

sr, scPSNR = model(imset)
print("PSNR score : ",scPSNR)

imsetshow(imset, k=5, figsize=(20,8), resample=False, show_histogram=True, show_map=True)

plt.figure(figsize=(30, 10))
plt.subplot(131);  plt.imshow(imset['lr'][0]);  plt.title('Low-Resolution-0 (300m / pixel)');
plt.subplot(132);  plt.imshow(sr);  plt.title('Super-Resolution (100m / pixel)');
plt.subplot(133);  plt.imshow(imset['hr']);  plt.title('Ground-truth high-resolution (100m / pixel)');
plt.show()
