import numpy as np
import cv2
import torch
from torch.utils import data
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
#from utils.utils import Get_Patch
import os
import math
import pickle
import random

class Dataset(data.Dataset):
    def __init__(self, list_IDs, labels, cfg, is_patch = True):
        'Initialization'
        
        self.list_IDs = list_IDs
        self.labels = labels
        self.cfg = cfg
            
        self.normalize = transforms.Normalize(cfg.mean, cfg.std)
        self.is_patch = is_patch
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        im_size = self.cfg.im_size
        image = cv2.imread(self.list_IDs[index],1)
        image = cv2.resize(image, (im_size,im_size))
        image = image/255.0
        image = np.moveaxis(image, 2, 0)
        image = np.float32(image)
        image = torch.from_numpy(image)
        image = self.normalize(image)
    
        mask = cv2.imread(self.labels[index],0)
        mask = cv2.resize(mask, (im_size,im_size), interpolation = cv2.INTER_NEAREST)
        mask = mask/255.0
        mask = torch.from_numpy(mask)
        
        return image, mask

def create_list(data,root): 

    img=[]
    msk=[]

    for value in data:
        value = value.split(" ")
        img.append(root+value[0])
        msk.append(root+value[1])

    return img,msk

#def get_file_names(cfg):
def get_file_names(cfg):

    imagenames = cfg.NIST_dir
    #imagenames = '/home/data/forgery/NC2016_Test0613.SCI/NIST_images'

    with open(imagenames, "rb") as fp: 
        files = pickle.load(fp)

    random.shuffle(files)
    root = imagenames[:-11]
    
    test_percent = 10
    test_len= int( float(len(files))  * (test_percent/100)  )
    
    test_list = files[-test_len:]
    train_list = files[:len(files)-test_len]

    test_img,test_msk = create_list(test_list,root)
    train_img, train_msk = create_list(train_list,root)


    train_IDs, mask_IDs = dict(list(enumerate(train_img))), dict(list(enumerate(train_msk)))
    test_IDs, mask_test_IDs = dict(list(enumerate(test_img))), dict(list(enumerate(test_msk)))


    return train_IDs, mask_IDs, test_IDs, mask_test_IDs

    #print(len(test_list), len(train_list))
   


class generator():
    def __init__(self,cfg):
        self.cfg = cfg
        self.train_IDs, self.mask_IDs, self.test_IDs, self.mask_test_IDs = get_file_names(cfg)

    def get_train_generator(self):
        
        batch_size = self.cfg.batch_size
        params = {'batch_size': batch_size,
                    'shuffle': True,
                    'pin_memory':True,
                    'num_workers': 4}
        training_set = Dataset(self.train_IDs, self.mask_IDs, self.cfg, is_patch=True)
        training_generator = data.DataLoader(training_set, **params)

        return training_generator

    def get_val_generator(self):

        batch_size = self.cfg.batch_size
        params = {'batch_size': batch_size,
                    'shuffle': False,
                    'pin_memory':True,
                    'num_workers': 4}
        

        val_set = Dataset(self.test_IDs, self.mask_test_IDs, self.cfg, is_patch = False)
        validation_generator = data.DataLoader(val_set, **params)
        return validation_generator

