import argparse
import typing
import io
import os

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import matplotlib.pyplot as plt

from urllib.request import urlretrieve

from PIL import Image
from torchvision import transforms

from models.modeling import VisionTransformer, CONFIGS

from datasets import build_dataset
import seaborn as sns

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

import time


def traversalDir_FirstDir(path):

    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            # print(m)
            if (os.path.isdir(m)):
                h = os.path.split(m)
                # print(h[1])
                list.append(h[1])
        # print(list)
    return list

def get_filename(path, filetype):  # path and file type, e.g.,'.csv'
    name = []
    for root,dirs,files in os.walk(path):
        for i in files:
            if os.path.splitext(i)[1]==filetype:
                name.append(i)    
    return name           


def token_rank(input_arr, name, path, block_num):
    # print('input arr is {}'.format(input_arr))
    ranked_token_value_idx = np.argsort(-input_arr) # ranked_avg_token_value_idx: each element M(i) indicates the token with the ranking (i)
    # print('sorted idx is {}'.format(ranked_token_value_idx))
    
    np.savetxt(path + '/{}_{}.txt'.format(name, block_num), ranked_token_value_idx,  fmt='%d')

path = '/data/yifan/ViT'
folder_name_list = traversalDir_FirstDir(path) # folder name is the label of the image

for folder_name in folder_name_list: # folder name e.g., /data/yifan/ViT/sea_slug, nudibranch
    print(folder_name)
    image_idx_list = traversalDir_FirstDir(path + '/' + folder_name)
    for image_idx in image_idx_list:
        print('img idx is {}'.format(image_idx))
        attention_path = path + '/' + folder_name + '/' + image_idx + '/attention_value/'
        attention_file_name_list = get_filename(attention_path, '.npy')
        attention_list = []
        if not os.path.isdir(path + '/' + folder_name + '/' + image_idx + '/token_value/'):
            os.mkdir(path + '/' + folder_name + '/' + image_idx + '/token_value/')
        
        if not os.path.isdir(path + '/' + folder_name + '/' + image_idx + '/token_value/token_value_1'):
            os.mkdir(path + '/' + folder_name + '/' + image_idx + '/token_value/token_value_1')

        if not os.path.isdir(path + '/' + folder_name + '/' + image_idx + '/token_value/token_value_2'):
            os.mkdir(path + '/' + folder_name + '/' + image_idx + '/token_value/token_value_2')

        if not os.path.isdir(path + '/' + folder_name + '/' + image_idx + '/token_value/token_value_sum'):
            os.mkdir(path + '/' + folder_name + '/' + image_idx + '/token_value/token_value_sum')

        if not os.path.isdir(path + '/' + folder_name + '/' + image_idx + '/token_value/token_value_2minus1'):
            os.mkdir(path + '/' + folder_name + '/' + image_idx + '/token_value/token_value_2minus1')

        for attention_file in attention_file_name_list:
            attention_value = np.load(attention_path + attention_file)
            attention_list.append(attention_value)
            avg_attention = np.mean(attention_value, axis = 0) # avg the attn across the attention heads
            token_value_1 = np.sum(avg_attention, axis = 1)[1:]
            token_value_2 = np.sum(avg_attention, axis = 0)[1:]
            token_value_sum = token_value_1 + token_value_2
            token_value_1minus2 = token_value_1 - token_value_2
            token_value_2minus1 = token_value_2 - token_value_1
            block_num = attention_file.replace('.npy', '')
            token_rank(token_value_1, 'token_value_1', path + '/' + folder_name + '/' + image_idx + '/token_value/token_value_1', block_num)
            token_rank(token_value_2, 'token_value_2', path + '/' + folder_name + '/' + image_idx + '/token_value/token_value_2', block_num)
            token_rank(token_value_sum, 'token_value_sum', path + '/' + folder_name + '/' + image_idx + '/token_value/token_value_sum', block_num)
            # token_rank(token_value_1minus2, 'token_value_1minus2', path + '/' + folder_name + '/' + image_idx + '/token_value/', block_num)
            token_rank(token_value_2minus1, 'token_value_2minus1', path + '/' + folder_name + '/' + image_idx + '/token_value/token_value_2minus1', block_num)

        attention_value  = np.stack(attention_list, axis=0) # attention value shape (12, 12, 197, 197), (layer, head, token #, token #) 
        attention_value = np.mean(attention_value, axis=1) # avg the attn across the attention heads
        attention_value = np.mean(attention_value, axis=0) # avg the attn across the layers


        avg_token_value_1 = np.sum(attention_value, axis = 1)[1:] # remove the cls_token, the sum of the elements in a row 
        avg_token_value_2 = np.sum(attention_value, axis = 0)[1:] # the sum of the elements in a column 
        avg_token_value_sum = avg_token_value_1 + avg_token_value_2
        avg_token_value_1minus2 = avg_token_value_1 - avg_token_value_2
        avg_token_value_2minus1 = avg_token_value_2 - avg_token_value_1

        token_rank(avg_token_value_1, 'token_value_1', path + '/' + folder_name + '/' + image_idx + '/token_value/', 'avg')
        token_rank(avg_token_value_2, 'token_value_2', path + '/' + folder_name + '/' + image_idx + '/token_value/', 'avg')
        token_rank(avg_token_value_sum, 'token_value_sum', path + '/' + folder_name + '/' + image_idx + '/token_value/', 'avg')
        # token_rank(avg_token_value_1minus2, 'avg_token_value_1minus2', path + '/' + folder_name + '/' + image_idx + '/token_value/', 'avg')
        token_rank(avg_token_value_2minus1, 'token_value_2minus1', path + '/' + folder_name + '/' + image_idx + '/token_value/', 'avg')