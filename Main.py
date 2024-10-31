import multiprocessing
from skimage.transform import rescale
import torch
import argparse
import faiss
import logging
import numpy as np
from tqdm import tqdm
from argparse import Namespace
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import os
import json
from time import time
import cv2
from PIL import Image, ImageFile, ImageDraw, ImageFont
from glob import glob
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sys
import torchvision
from torch import nn
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_att = self.channel_attention(x)
        x = x * channel_att

        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_att = torch.cat([max_pool, avg_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_att)
        x = x * spatial_att

        return x

class ResNetBottleneckAttention(nn.Module):
    def __init__(self, num_classes=512):
        super(ResNetBottleneckAttention, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.cbam1 = CBAM(256)

        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.cbam2 = CBAM(512)

        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.cbam3 = CBAM(1024)

        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.cbam4 = CBAM(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.cbam1(x)
        x = self.layer2(x)
        x = self.cbam2(x)
        x = self.layer3(x)
        x = self.cbam3(x)
        x = self.layer4(x)
        x = self.cbam4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def read_images_paths(dataset_folder, get_abs_path=False):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")

    file_with_paths = dataset_folder + "_images_paths.txt"
    if os.path.exists(file_with_paths):
        logging.debug(f"Reading paths of images within {dataset_folder} from {file_with_paths}")
        with open(file_with_paths, "r") as file:
            images_paths = file.read().splitlines()
        images_paths = [os.path.join(dataset_folder, path) for path in images_paths]
        # Sanity check that paths within the file exist
        if not os.path.exists(images_paths[0]):
            raise FileNotFoundError(f"Image with path {images_paths[0]} "
                                    f"does not exist within {dataset_folder}. It is likely "
                                    f"that the content of {file_with_paths} is wrong.")
    else:
        logging.debug(f"Searching images in {dataset_folder} with glob()")
        images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
        if len(images_paths) == 0:
            raise FileNotFoundError(f"Directory {dataset_folder} does not contain any JPEG images")

    if not get_abs_path:  # Remove dataset_folder from the path
        images_paths = [p[len(dataset_folder) + 1:] for p in images_paths]

    return images_paths




def writer_txt(position_list):
    with open("new_fanben.txt", 'r', encoding='utf-8') as f:
        json1 = json.load(f)
        posi_list = json1['provinces'][0]['cities']
        j = 0
        for i in position_list:
            json1['provinces'][0]['cities'][j]['g'] = str(position_list[j][0]) + ',' + str(position_list[j][1]) + '|12'
            j += 1
        json2 = json.dumps(json1, ensure_ascii=False)  # dumps 默认会以ASCII码，关掉ensure_ascii=False后原汁原味转json
        open(r"C:\Users\Administrator\Desktop\new.txt", "w", encoding='utf-8').write(json2)
        print('已写入完成')


if __name__ =='__main__':
    # RECALL_VALUES = [1, 5, 10, 20]
    torch.backends.cudnn.benchmark = True
    args = parse_arguments()
    model = ResNetBottleneckAttention(512)
    model_state_dict = torch.load(
         r"E:\paper_data_procedure\picture_location\amend_location\logs\default\2024-09-25_11-05-55\best_model.pth")
    model.load_state_dict(model_state_dict)
    
    total_params = 0
    for name, param in model.named_parameters():
        # if param.requires_grad:  # 只计算需要梯度的参数（即可训练的参数）
        param_num = param.numel()  # 获取参数的元素数量
        total_params += param_num

    print(f"Total number of parameters: {total_params}")

    # export_model_onnx(model)   # 导出模型

    model = model.to(args.device)
   
    test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1",resize_test_imgs=args.resize_test_imgs,
                          positive_dist_threshold=args.positive_dist_threshold)
    simi, predictions  = test(args, test_ds, model, args.num_preds_to_save)
    # new_pre = np.reshape(predictions,(-1))
    # new_pre_1 = list(new_pre)
    # print(new_pre_1)
    # j = 0
    # posi_list = []
    # for i in new_pre_1:
    #     # query = test_ds.queries_paths[j].split('\\')
    #     # query = (query[6].split('.'))[0]
    #     # database = test_ds.database_paths[i].split('@')
    #     # longitude, latitude = database[6],database [5]
    #     # print(f'图片{query}的经度为{longitude}，纬度为{latitude}')
    #
    #     # print(i)
    #     database = test_ds.database_paths[i].split('@')
    #     longitude, latitude = database[6], database[5]
    #     print(f'图片{test_ds.queries_paths[j]}的经度为{longitude}，纬度为{latitude},最接近的图片为{test_ds.database_paths[i]}')
    #     posi_list.append([longitude, latitude])
    #     plot_pic(test_ds.queries_paths[j],test_ds.database_paths[i])
    #     j += 1
    # # writer_txt(posi_list)
    # # print(simi)     #  distance
    # # print(predictions)    #  conresponding_img
    print(simi,predictions)

    posi_list = []
    print('预测查询长度',len(predictions))
    for i in range(len(predictions)):
        print('查询图片：',test_ds.queries_paths[i])
        print("对应卫星图：",test_ds.database_paths[predictions[i][0]])
        plot_pic2(test_ds.queries_paths[i], test_ds.database_paths[predictions[i][0]],test_ds.database_paths[
            predictions[i][1]])

