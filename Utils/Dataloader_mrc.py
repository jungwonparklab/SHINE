from __future__ import print_function, division
import os
import torch
import numpy as np
import random
import cv2 as cv
import numba
import bisect
import mrcfile
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
import torchvision.transforms.functional as tvF
import torchvision.transforms.v2 as T
from torchvision.io import read_image
from torchvision.transforms.transforms import ToTensor
from Utils.Utils import *
from PIL import Image
        
def imageloader(path,index,total_length,list):
    image_path = os.path.join(path,list[index])
    image_in = torch.Tensor((cv.imread(image_path,flags=cv.IMREAD_UNCHANGED).astype(np.float32))).unsqueeze(0)
    #image_in = torch.Tensor((Image.open(image_path).astype(np.float32))).unsqueeze(0)
    
    return image_in

def Sequentialloader(image_dir, image_size,gt_path=None,validation_length=1,recursive_factor=10,frame_num=5,noise_scaler=0.25):
    total_length = len(os.listdir(image_dir))
    training_length = total_length-validation_length
    image_size = image_size
    index = random.sample(list(range(0, total_length)), total_length)
    t_index = index[0:training_length]
    v_index = index[training_length:training_length+validation_length]
    if gt_path is None:
        gt_path = image_dir
    Trainset = TrainLoader(image_dir, training_length, gt_path,total_length,image_size,t_index,recursive_factor,noise_scaler)
    Validationset = ValidationLoader(image_dir, validation_length, gt_path,total_length,image_size,v_index)
    #transforms = torch.jit.script(transforms)
    return Trainset, Validationset


@numba.jit(nopython=True, cache=True)
def numpy_normalize(image):
    output = np.empty_like(image)
    channels = image.shape[0]
    for c in range(channels):
        v = image[c,:,:].flatten()
        max = v.max()
        min = v.min()
        output[c,:,:] = (image[c,:,:] - min) / (max-min)
    return output

@numba.jit(nopython=True, cache=True)
def numpy_zscore_normalize(image):
    output = np.empty_like(image)
    channels = image.shape[0]
    for c in range(channels):
        v = image[c,:,:].flatten()
        mean = v.mean()
        std = v.std()
        output[c,:,:] = (image[c,:,:] - mean) / std
    return output

@numba.jit(nopython=True, cache=True)
def shuffle_blocks(image, kernel):
    c, h, w = image.shape
    new_image = np.empty_like(image)
    for i in range(0, h, kernel):
        for j in range(0, w, kernel):
            block = image[:, i:i+kernel, j:j+kernel]
            contiguous_block = np.ascontiguousarray(block)
            flat_block = contiguous_block.reshape(c, -1)
            num_pixels = flat_block.shape[1]
            pixel_indices = np.arange(num_pixels)
            np.random.shuffle(pixel_indices)
            shuffled_block = flat_block[:, pixel_indices]
            reshaped_block = shuffled_block.reshape(c, int(kernel), int(kernel))  # Explicitly cast to integers
            new_image[:, i:i+kernel, j:j+kernel] = reshaped_block
    return new_image

@numba.jit(nopython=True, cache=True)
def clip_top_3_percent(img):
    assert img.ndim == 3, "Input must be a 3D array"
    c, h, w = img.shape
    clipped_img = np.zeros_like(img)
    for i in range(c):
        channel = img[i, :, :]
        threshold = np.percentile(channel, 99.7)
        clipped_img[i, :, :] = np.clip(channel, None, threshold)
    return clipped_img

@torch.jit.script
def gauss_noise_torch(img, noise_scaler: float = 0.25):
    #img = torch_zscore_normalize(img)
    
    # sigma is now a scalar tensor
    sigma = torch.rand(1) * noise_scaler + 0.25

    # Generate Gaussian noise
    noise = sigma * torch.randn_like(img)

    # Add noise to the image
    out = img + noise
    out = torch_zscore_normalize(out)
    #out = torch.clamp(out, 0, 1)

    return out

class TrainLoader(Dataset):
    def __init__(self, image_dir, training_length, gt_path,total_length,image_size,index,recursive_factor,noise_scaler=0.25):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.total_length = len(os.listdir(self.image_dir))
        self.training_length = training_length
        self.recursive_factor = recursive_factor
        self.transforms = T.Compose([
        T.RandomResize(256,1024,interpolation=T.InterpolationMode.NEAREST),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(0.50),
        T.RandomVerticalFlip(0.5),
        T.RandomApply([T.RandomRotation((90, 90))], 0.5),
        #T.ConvertImageDtype(torch.float),
        #T.Normalize([0], [1])
        ])
        self.gt_path = gt_path
        self.datalength = total_length
        self.index = index 
        self.noise_scaler = noise_scaler

    def __len__(self):
        return int(self.training_length*self.recursive_factor)


    def get_mean_std(self):
        mean = 0.
        std = 0.
        self.mean = mean
        self.std = std
        print('mean: ', mean)
        print('std: ', std)
        return mean, std, mean

    def normalize(self, tensor):
        return (tensor - self.mean) / self.std

    def __getitem__(self, idx):
        idx = idx%self.training_length
        idxhat = self.index[idx]
        image_path = os.path.join(self.image_dir, self.image_list[idxhat])
        batch_image = np.load(image_path)['data']
        batch_image = (batch_image).astype(np.float32)
        batch_processed = (batch_image)
        batch_processed = torch.from_numpy(batch_processed)
        batch_processed = batch_processed
        batch_processed = (torch_zscore_normalize(self.transforms(batch_processed)))
        return batch_processed

class ValidationLoader(Dataset):
    def __init__(self,image_dir, validation_length, gt_path,total_length,image_size,index):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.total_length = total_length
        self.validation_length = validation_length
        self.transforms = T.Compose([
        T.CenterCrop(image_size),
        #T.ConvertImageDtype(torch.float),
        #T.Normalize([0], [1])
        ])
        self.gt_path = gt_path
        #self.gt_list = sorted(os.listdir(gt_path))
        self.index = index

    def __len__(self):
        return int(self.validation_length)        

    def __getitem__(self, idx):
        val_index = self.index[idx]
        image_path = os.path.join(self.image_dir, self.image_list[val_index])
        batch_image = torch.Tensor((np.load(image_path)['data']))
        batch_image = batch_image
        batch_processed = self.transforms(batch_image)
        batch_processed = torch_zscore_normalize(batch_processed)
        return batch_processed

class TestLoader_large(Dataset):
    def __init__(self,image_dir,subset=None,frame_num=5):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.total_length = len(os.listdir(image_dir))
        self.frame_num = frame_num
        self.transforms = T.Compose([
        T.ConvertImageDtype(torch.float),
        #T.Normalize([0], [1])
        ])
        if subset is not None:
            self.subset=subset
        else:
            self.subset=self.total_length

    def __len__(self):
        return int(self.total_length)       

    def __getitem__(self, idx): 
        idx_list = idxreturn(idx,self.subset,frame_num=self.frame_num)
        first=True
        for idx in idx_list:
            img = imageloader(self.image_dir,idx,self.total_length,self.image_list)
            if first:
                first=False
                img_series = img
                img_name = self.image_list[idx_list[self.frame_num//2]]
            else:
                img_series = np.concatenate((img_series, img),0)
        batch_image = img_series
        batch_processed = self.transforms(batch_image)
        return batch_processed, idx, img_name

class TestLoader_mrc(Dataset):
    def __init__(self,image_dir,subset=None,gain_dir=None):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.total_length = len(os.listdir(image_dir))
        if subset is not None:
            self.subset=subset
        else:
            self.subset=self.total_length
        if isinstance(gain_dir, type(None)):
            self.gain_value = None
        else:
            gain = mrcfile.mmap(gain_dir, mode='r')
            gain_value = gain.data 
            gain_value = np.flipud(gain_value)
            self.gain_value = torch.tensor(gain_value.copy())

    def __len__(self):
        return int(self.total_length)       

    def __getitem__(self, idx):
        im0 = mrcfile.open(os.path.join(self.image_dir,self.image_list[idx]))
        image = torch.tensor(np.array(im0.data,dtype=np.float32))
        if isinstance(self.gain_value, type(None)):
            gain_value = torch.ones_like(image[0,:,:])
        else:
            gain_value = self.gain_value
        data = image*gain_value
        img_name = self.image_list[idx]


        return data, idx, img_name, gain_value

import ncempy.io.dm as dm

class TestLoader_dm4(Dataset):
    def __init__(self, image_dir, subset=None, gain_dir=None, frames = 5):
        self.image_dir = image_dir
        self.frame_num = frames
        self.image_list = sorted(os.listdir(image_dir))
        self.total_length = 0
        self.file_loadnumber = []
        self.file_indexes = []
        z = 0
        for file_num in range(0,len(self.image_list)):
            image_stack = dm.fileDM(os.path.join(self.image_dir,self.image_list[file_num]))
            stacks1, stacks2, width, height = image_stack.getMemmap(0).shape
            self.total_length += stacks1 * stacks2
            for s in range(0, stacks1 * stacks2):
                self.file_loadnumber.append(file_num)
                idx_list = idxreturn(s, stacks1*stacks2, frames)
                self.file_indexes.append(idx_list)
                z = z+1
        if subset is not None:
            self.subset = subset
        else:
            self.subset = self.total_length
        if isinstance(gain_dir, type(None)):
            self.gain_value = None
        else:
            gain = dm.fileDM(gain_dir)
            gain_value = gain.getDataset(0).astype(np.float32)
            gain_value = np.flipud(gain_value)
            self.gain_value = torch.tensor(gain_value.copy())

    def __len__(self):
        return int(self.total_length)

    def __getitem__(self, idx):
        file_num = self.file_loadnumber[idx]
        file_indexes = self.file_indexes[idx]
        im0 = dm.fileDM(os.path.join(self.image_dir,self.image_list[file_num]))
        stacks1, stacks2, width, height = im0.getMemmap(0).shape
        first= True
        for idx in file_indexes:
            s1 = idx//stacks2
            s2 = idx%stacks2
            image_data = torch.tensor(im0.getMemmap(0)[s1,s2,:,:].copy().astype(np.float32)).unsqueeze(0)
            if first:
                first=False
                batch_image = image_data
            else:
                batch_image = torch.cat((batch_image,image_data),dim=0)
        if isinstance(self.gain_value, type(None)):
            gain_value = torch.ones((batch_image.shape[-2],batch_image.shape[-1]))
        else:
            gain_value = self.gain_value
        img_name = os.path.basename(self.image_list[file_num])

        return batch_image, file_indexes[self.frame_num//2], img_name, gain_value

def Sequentialloader_single(image_dir, image_size,gt_path=None,validation_length=1,recursive_factor=10):
    total_length = len(os.listdir(image_dir))
    training_length = total_length-validation_length
    image_size = image_size
    index = random.sample(list(range(0, total_length)), total_length)
    t_index = index[0:training_length]
    v_index = index[training_length:training_length+validation_length]
    if gt_path is None:
        gt_path = image_dir
    Trainset = TrainLoader_single(image_dir, training_length, gt_path,total_length,image_size,t_index,recursive_factor)
    Validationset = ValidationLoader_single(image_dir, validation_length, gt_path,total_length,image_size,v_index)
    #transforms = torch.jit.script(transforms)
    return Trainset, Validationset


class TrainLoader_single(Dataset):
    def __init__(self, image_dir, training_length, gt_path,total_length,image_size,index,recursive_factor):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.total_length = len(os.listdir(self.image_dir))
        self.training_length = training_length
        self.recursive_factor = recursive_factor
        self.transforms = T.Compose([
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(0.50),
        T.RandomVerticalFlip(0.5),
        T.RandomApply([T.RandomRotation((90, 90))], 0.5),
        #T.ConvertImageDtype(torch.float),
        #T.Normalize([0], [1])
        ])
        self.gt_path = gt_path
        self.gt_list = sorted(os.listdir(gt_path))
        self.datalength = total_length
        self.index = index  

    def __len__(self):
        return int(self.training_length*self.recursive_factor)

    def __getitem__(self, idx):
        idx = idx%self.training_length
        idxhat = self.index[idx]
        image_path = os.path.join(self.image_dir, self.image_list[idxhat])
        batch_image = torch.Tensor((np.load(image_path)['data']))
        if len(batch_image.size())==3: #if image is grayscale
            batch_image = batch_image.unsqueeze(1)
        batch_processed = self.transforms(batch_image)
        batch_processed = numpy_zscore_normalize(batch_processed)
        previous_out = batch_processed[0, :, :]
        return previous_out,previous_out

class ValidationLoader_single(Dataset):
    def __init__(self,image_dir, validation_length, gt_path,total_length,image_size,index):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.total_length = total_length
        self.validation_length = validation_length
        self.transforms = T.Compose([
        T.CenterCrop(image_size),
        #T.ConvertImageDtype(torch.float),
        #T.Normalize([0], [1])
        ])
        self.gt_path = gt_path
        self.gt_list = sorted(os.listdir(gt_path))
        self.index = index

    def __len__(self):
        return int(self.validation_length)        

    def __getitem__(self, idx):
        val_index = self.index[idx]
        image_path = os.path.join(self.image_dir, self.image_list[val_index])
        batch_image = torch.Tensor((np.load(image_path)['data']))
        if len(batch_image.size())==3: #if image is grayscale
            batch_image = batch_image.unsqueeze(1)
        batch_processed = self.transforms(batch_image)
        batch_processed = numpy_zscore_normalize(batch_processed)
        previous_out = batch_processed[0, :, :]
        return previous_out,previous_out

class TestLoader_single(Dataset):
    def __init__(self,image_dir,subset=None):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.total_length = len(os.listdir(image_dir))
        #self.transforms = T.Compose([
        #T.ConvertImageDtype(torch.float),
        #T.Normalize([0], [1])
        #])
        if subset is not None:
            self.subset=subset
        else:
            self.subset=self.total_length

    def __len__(self):
        return int(self.total_length)       

    def __getitem__(self, idx): 
        previous_in = imageloader(self.image_dir,idx,self.total_length,self.image_list)
        img_name = self.image_list[idx]
        batch_image = previous_in.unsqueeze(0)
        batch_processed = batch_image
        #batch_processed = self.transforms(batch_image)
        previous_out = batch_processed[0, :, :]
        return previous_out, idx, img_name