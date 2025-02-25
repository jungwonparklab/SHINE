import os
import torch
import numpy as np
import random
import cv2 as cv    
from PIL import Image
from torch.utils.data import Dataset
from Utils.Utils import *
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
from functools import cache
import numba
py_totensor = T.ToTensor()

def imageloader(image_path):
    #return cv.imread(image_path, flags=cv.IMREAD_UNCHANGED).astype(np.float32)
    return np.asarray(Image.open(image_path)).astype(np.float32)

def Sequentialloader_plain(image_dir, image_size,gt_path=None,validation_length=1,recursive_factor=1,frame_num=5):
    total_length = len(os.listdir(image_dir))
    training_length = total_length-validation_length
    image_size = image_size
    index = random.sample(list(range(0, total_length)), total_length)
    t_index = index[0:training_length]
    v_index = index[training_length:training_length+validation_length]
    
    Trainset = TrainLoader_plain(image_dir, training_length, gt_path,total_length,image_size,t_index,recursive_factor,frame_num)
    Validationset = ValidationLoader_plain(image_dir, validation_length, gt_path,total_length,image_size,v_index,frame_num)
    #transforms = torch.jit.script(transforms)
    return Trainset, Validationset

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

@torch.jit.script
def gauss_noise_torch(img: torch.Tensor) -> torch.Tensor:
    dtype = img.dtype
    #img = torch_zscore_normalize(img)
    minimum = img.min()
    maximuum = img.max()
    
    # sigma is now a scalar tensor
    sigma = torch.rand(1) * 0.5 + 0.25

    # Generate Gaussian noise
    noise = sigma * torch.randn_like(img)

    # Add noise to the image
    out = img + noise
    #out = torch.clamp(out, minimum)
    out = torch_zscore_normalize(out)

    return out

class TrainLoader_plain(Dataset):
    def __init__(self, image_dir, training_length, gt_path,total_length,image_size,index,recursive_factor,frame_num):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.total_length = len(os.listdir(self.image_dir))
        self.training_length = training_length
        self.transforms = T2.Compose([
        T2.RandomResize(256,512,interpolation=T2.InterpolationMode.NEAREST),
        T2.RandomCrop(image_size),
        T2.RandomHorizontalFlip(0.50),
        T2.RandomVerticalFlip(0.5)]
        #T2.RandomApply([T2.RandomRotation((90, 90))], 0.5)]
        )
        self.gt_path = gt_path
        self.gt_list = sorted(os.listdir(gt_path))
        self.full_image_paths = [os.path.join(self.image_dir, img_name) for img_name in self.image_list]
        self.datalength = total_length
        self.index = index  
        self.recursive_factor =recursive_factor
        self.frame_num = frame_num
        sample_image = imageloader(self.full_image_paths[0])
        h,w = sample_image.shape
        self.image_size_h = h
        self.image_size_w = w

    def __len__(self):
        return int(self.training_length*self.recursive_factor)
    
    def get_mean_std(self):
        mean = 0
        std = 0
        maximum = 0
        for i in range(self.training_length):
            img = imageloader(self.full_image_paths[i]).flatten().astype(np.float32)
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
    
    def normalize(self, img):
        img = (img - self.mean) / self.std
        return img

    def __getitem__(self, idx):
        idx = idx % self.training_length
        train_idx = self.index[idx]

        idxlist = idxreturn(train_idx, self.total_length, self.frame_num)

        # Preallocate array
        images_array = np.zeros((self.frame_num, self.image_size_h,  self.image_size_w), dtype=np.float32)

        for i, idx in enumerate(idxlist):
            images_array[i] = imageloader(self.full_image_paths[idx])

        # Concatenate all the images in one step
        #batch_image = np.concatenate(images_array, axis=0)
        batch_image = images_array
        batch_processed = gauss_noise_torch(torch_zscore_normalize(self.transforms(torch.tensor(batch_image.astype(np.float32)))))
        batches = batch_processed[:, :, :]
        gt_out = batch_processed[self.frame_num//2, :, :]
        return batches

class ValidationLoader_plain(Dataset):
    def __init__(self,image_dir, validation_length, gt_path,total_length,image_size,index,frame_num):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.total_length = total_length
        self.validation_length = validation_length
        self.transforms = T.Compose([
        T2.FiveCrop(image_size)
        ])
        self.gt_path = gt_path
        self.gt_list = sorted(os.listdir(gt_path))
        self.full_image_paths = [os.path.join(self.image_dir, img_name) for img_name in self.image_list]
        self.index = index
        self.frame_num = frame_num

        sample_image = imageloader(self.full_image_paths[0])
        h,w = sample_image.shape
        self.image_size_h = h
        self.image_size_w = w
                                   
    def __len__(self):
        return int(self.validation_length) * 4 * 5

    def __getitem__(self, idx):
        idx_img = idx//20
        rot = idx%4
        crop = idx%5
        val_index = self.index[idx_img]
        idxlist = idxreturn(val_index,self.total_length,self.frame_num)
        num_images = len(idxlist)

        # Preallocate array
        images_array = np.zeros((self.frame_num, self.image_size_h,  self.image_size_w), dtype=np.float32)

        for i, idx in enumerate(idxlist):
            images_array[i] = imageloader(self.full_image_paths[idx])
        # Concatenate all the images in one step
        #batch_image = np.concatenate(images_array, axis=0)
        batch_processed = self.transforms(torch.tensor((images_array.astype(np.float32))))
        batch_processed = batch_processed[crop]
        batch_processed = torch_zscore_normalize(batch_processed)
        #batch_processed = torch.rot90(batch_processed,rot,[-2,-1])
        batches = batch_processed[:, :, :]
        gt_out = batch_processed[self.frame_num//2, :, :]
        coin=[0, 1]
        p = random.choice(coin)
        if p==0:
            batches=batches
        else:
            batches=torch.flip(batches,dims=[0])
        return batches

class TestLoader_plain(Dataset):
    def __init__(self,image_dir,frame_num=5):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.total_length = len(os.listdir(image_dir))
        self.frame_num = frame_num
        self.full_image_paths = [os.path.join(self.image_dir, img_name) for img_name in self.image_list]
        self.transforms = T2.Compose([
        T2.ConvertImageDtype(torch.float),
        ])

        sample_image = imageloader(self.full_image_paths[0])
        h,w = sample_image.shape
        self.image_size_h = h
        self.image_size_w = w


    def __len__(self):
        return int(self.total_length)        

    def __getitem__(self, idx):
        idxlist = idxreturn(idx,self.total_length,self.frame_num)
        img_name = self.image_list[idxlist[self.frame_num//2]]
        # Preallocate array
        images_array = np.zeros((self.frame_num, self.image_size_h,  self.image_size_w), dtype=np.float32)

        for i, idx in enumerate(idxlist):
            images_array[i] = imageloader(self.full_image_paths[idx])

        # Concatenate all the images in one step
        #batch_image = np.concatenate(images_array, axis=0)
        batch_processed = (self.transforms(torch.tensor(images_array.astype(np.float32))))
        return batch_processed, idx, img_name
