from __future__ import print_function, division
import os
import torch
import numpy as np
import random
import cv2 as cv
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tvF
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.transforms.transforms import ToTensor

random.seed(a=None,version=22)

def numpy_normalize(image):
    if np.issubdtype(image.dtype, np.floating):
        v = image[:,1]
        image =(image.astype(np.float64)-v.min())/(v.max()-v.min())
    else:
        v = image[:,1]
        image = image.astype(np.float64)/v.max()
    return image

def imageloader(path,index,total_length,list):
    image_path = os.path.join(path,list[index])
    image_in = torch.Tensor(numpy_normalize(cv.imread(image_path,flags=cv.IMREAD_UNCHANGED))).unsqueeze(0)
    return image_in

##mask_generator for N2V
def generate_mask(input, ratio = 1-0.00198,size_window=[5, 5]):
    ratio = ratio
    size_window = size_window
    size_data = input.shape
    num_sample = int(size_data[1] * size_data[2] * (1 - ratio))

    mask = np.ones(size_data)
    output = input

    for ich in range(size_data[0]):
        idy_msk = np.random.randint(0, size_data[1], num_sample)
        idx_msk = np.random.randint(0, size_data[2], num_sample)

        idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, num_sample)
        idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, num_sample)

        idy_msk_neigh = idy_msk + idy_neigh
        idx_msk_neigh = idx_msk + idx_neigh

        idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[1] - (idy_msk_neigh >= size_data[1]) * size_data[1]
        idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[2] - (idx_msk_neigh >= size_data[2]) * size_data[2]

        id_msk = (ich, idy_msk, idx_msk)
        id_msk_neigh = (ich, idy_msk_neigh, idx_msk_neigh)

        output[id_msk] = input[id_msk_neigh]
        mask[id_msk] = 0.0

    return output, mask

def Sequentialloader_N2V(image_dir, image_size,gt_path=None,validation_length=1,recursive_factor=1):
    total_length = len(os.listdir(image_dir))
    training_length = total_length-validation_length
    image_size = image_size
    index = random.sample(list(range(0, total_length)), total_length)
    t_index = index[0:training_length]
    v_index = index[training_length:training_length+validation_length]
    
    Trainset = TrainLoader_N2V(image_dir, training_length, gt_path,total_length,image_size,t_index,recursive_factor)
    Validationset = ValidationLoader_N2V(image_dir, validation_length, gt_path,total_length,image_size,v_index)
    #transforms = torch.jit.script(transforms)
    return Trainset, Validationset

class TrainLoader_N2V(Dataset):
    def __init__(self, image_dir, training_length, gt_path,total_length,image_size,index,recursive_factor):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.total_length = len(os.listdir(self.image_dir))
        self.training_length = training_length
        self.transforms = T.Compose([
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(0.50),
        T.RandomVerticalFlip(0.5),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0], [1])
        ])
        self.gt_path = gt_path
        self.gt_list = sorted(os.listdir(gt_path))
        self.datalength = total_length
        self.index = index
        self.recursive_factor=recursive_factor

    def __len__(self):
        return int(self.training_length*self.recursive_factor)

    def get_mean_std(self):
        mean = 0
        std = 0
        maximum = 0
        for i in range(self.training_length):
            img = np.array(imageloader(self.image_dir,i,self.training_length,self.image_list).flatten()).astype(np.float32)
            mean += img.mean()
            std += img.std()
            maximum = np.maximum(maximum, img.max())
        mean /= self.training_length
        std /= self.training_length
        self.mean = mean
        self.std = std
        print('mean: ', mean)
        print('std: ', std)
        return mean/maximum , std/maximum, maximum

    def __getitem__(self, idx):
        idx = idx%self.training_length
        now_in = imageloader(self.image_dir,idx,self.training_length,self.image_list)
        gt_in = imageloader(self.gt_path,idx,self.training_length,self.gt_list)
        batch_image = torch.cat((now_in.unsqueeze(0),gt_in.unsqueeze(0)), dim=0)
        batch_processed = self.transforms(batch_image)
        now_out = batch_processed[0, :, :]
        gt_out = batch_processed[1, :, :]
        now, mask = generate_mask(now_out)
        return now, mask

class ValidationLoader_N2V(Dataset):
    def __init__(self,image_dir, validation_length, gt_path,total_length,image_size,index):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.total_length = total_length
        self.validation_length = validation_length
        self.transforms = T.Compose([
        T.RandomCrop(image_size),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0], [1])
        ])
        self.gt_path = gt_path
        self.gt_list = sorted(os.listdir(gt_path))
        self.index = index

    def __len__(self):
        return int(self.validation_length)        

    def __getitem__(self, idx):
        idxhat = -idx
        now_in = imageloader(self.image_dir,idxhat,self.total_length,self.image_list)
        gt_in = imageloader(self.gt_path,idxhat,self.total_length,self.gt_list)
        batch_image = torch.cat((now_in.unsqueeze(0),gt_in.unsqueeze(0)), dim=0)
        batch_processed = self.transforms(batch_image)
        now_out = batch_processed[0, :, :]
        gt_out = batch_processed[1, :, :]
        now, mask = generate_mask(now_out)
        return now, mask

class TestLoader(Dataset):
    def __init__(self,image_dir):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.total_length = len(os.listdir(image_dir))
        self.transforms = T.Compose([
        T.ConvertImageDtype(torch.float),
        T.Normalize([0], [1])
        ])

    def __len__(self):
        return int(self.total_length)        

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        now_in = imageloader(self.image_dir,idx,self.total_length,self.image_list)
        batch_image = now_in.unsqueeze(0)
        batch_processed = self.transforms(batch_image)
        now_out = batch_processed.squeeze(0)
        if not now_out.shape[2]//32==0:
            cropsize = 32*(now_out.shape[2]//32)
            now_out = tvF.center_crop(now_out, cropsize)
        return now_out, idx, img_name