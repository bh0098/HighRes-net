import json
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import torch
from DeepNetworks.HRNet import HRNet
import argparse


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


def main(config,args):
    # config_file_path = "../config/config.json"
    # with open(config_file_path, "r") as read_file:
    #     config = json.load(read_file)
    # print(config)
    # lr_add = "../data/train/NIR/imgset0596/LR000.png"
    lr_add = args.lr_add
    hr_add = args.hr_add
    # hr_add = "../data/train/NIR/imgset0596/HR.png"
    lr_img = plt.imread(lr_add)
    hr_img = plt.imread(hr_add)
    # model address

    # transformers to make a image a tensor
    tfms = transforms.Compose([
        transforms.ToTensor()])

    # checkpoint_dir = "..\models\weights"
    # run_subfolder = 'batch_8_views_32_min_32_beta_50.0_time_2021-12-10-20-32-43-747510'
    # checkpoint_filename = 'HRNet.pth'
    # checkpoint_file = os.path.join(checkpoint_dir, run_subfolder, checkpoint_filename)
    # print(checkpoint_file)
    checkpoint_file = args.model
    assert os.path.isfile(checkpoint_file)
    model = load_model(config, checkpoint_file)

    model.eval()
    # mp_lr = np.ones(lr_img.shape)

    dev = 'cpu'
    model.to(dev)
    batch_lr_img = tfms(lr_img).unsqueeze(0)
    alpahs = torch.tensor(np.ones(batch_lr_img[0].size()))
    print(batch_lr_img.shape)
    out = model(tfms(lr_img).unsqueeze(0), alpahs.unsqueeze(0))
    sr_img = out[0][0].detach().numpy()
    # plt.imshow(sr_img.detach().numpy())

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 3, 1)
    # imgplot = plt.imshow(lr_img)
    # ax.set_title('low res')
    # ax = fig.add_subplot(1, 3, 2)
    # imgplot = plt.imshow(hr_img)
    # ax.set_title('high res')
    # ax = fig.add_subplot(1, 3, 3)
    # imgplot = plt.imshow(sr_img)
    # ax.set_title('super res')
    # plt.show()

    fig = plt.figure(figsize=(6,3),dpi=100)
    ax = fig.add_subplot(1,2,1)
    imgplot = plt.imshow(lr_img[:lr_img.shape[0]//3,:lr_img.shape[0]//3])
    ax.set_title('low res')
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(sr_img[:sr_img.shape[0]//3,:sr_img.shape[0]//3])
    ax.set_title('super res')
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path of the config file", default='config/config.json')
    parser.add_argument("--lr_add", help="path of low resolution img", default="data/train/NIR/imgset0596/LR000.png")
    parser.add_argument("--hr_add", help="path of high resolution img", default="data/train/NIR/imgset0596/HR.png")
    parser.add_argument("--model", help="path of trained model",
                        default="models/weights/batch_8_views_32_min_32_beta_50.0_time_2021-12-10-20-32-43-747510/HRNet.pth")

    args = parser.parse_args()
    print(args.config)
    assert os.path.isfile(args.config)

    with open(args.config, "r") as read_file:
        config = json.load(read_file)

    main(config,args)
