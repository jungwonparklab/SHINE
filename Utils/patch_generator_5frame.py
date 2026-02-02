# -*- coding: utf-8 -*-

import glob
import multiprocessing
import os
import numpy as np
import random
from multiprocessing import Pool, Manager
import mrcfile
import cv2 as cv
from tqdm import tqdm
import ncempy.io.dm as dm

def data_aug(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.fliplr(img)
    elif mode == 3:
        return np.flipud(np.fliplr(img))

def idxreturn(idx,stack_num,frame_num):
    half = frame_num//2
    idx_list = []
    if idx>(stack_num-half-1):
        for i in range(-half, half+1):
            idx_list.append(idx-abs(i))
        return idx_list

    if idx<half:
        for i in range(-half, half+1):
            idx_list.append(idx+abs(i))
        return idx_list
    else:
        for i in range(-half, half+1):
            idx_list.append(idx+i)
        return idx_list

def generate_patch_memory_eficient_gainfix(img_dir,gain_dir,save_dir,patch_size,stride,aug_times,processor_num=20, ratio=0.1, frame_num=5):
    patch_size, stride = patch_size, stride
    aug_times = aug_times
    save_dir = save_dir
    gain_dir = gain_dir
    src_dir = img_dir+'/'
    frames = frame_num
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(processes=processor_num)
    file_list = sorted(glob.glob(src_dir+'*.mrc'))
    input = []
    for file_num in range(0,len(file_list)):
        image_stack = mrcfile.mmap(file_list[file_num], mode='r')
        stacks = image_stack.data.shape[0]
        width = image_stack.data.shape[1]
        height = image_stack.data.shape[2]
        for s in range(0,stacks):
            idx_list = idxreturn(s,stacks,frames)
            input.append((file_list[file_num],gain_dir,idx_list,patch_size,stride,aug_times,save_dir,file_num,s,width,height,ratio))
    with tqdm(total=len(input)) as pbar:
        for _ in tqdm(pool.imap_unordered(map_function,input)):
            pbar.update()

    input =[]
    pool.close()
    pool.join()

    return print('finish')

def map_function(i):
    return gen_patches_with_gainfix(i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7],i[8],i[9],i[10],i[11])

def gen_patches_with_gainfix(file_name,gain_dir,idx_list,patch_size,stride,aug_times,save_dir,mrc_num,stack_num,width,height,ratio=0.1):
    image_stack = mrcfile.mmap(file_name, mode='r')
    if isinstance(gain_dir, type(None)):
        gain_value = np.ones((width,height))
    else:
        gain = mrcfile.mmap(gain_dir, mode='r')
        gain_value = gain.data 
        gain_value = np.flipud(gain_value)
    patches=[]
    for i in range(0, width-patch_size+1, stride):
        for j in range(0, height-patch_size+1, stride):
            if random.random()<ratio:
                first=True
                w_rand=0
                h_rand=0
                for idx in idx_list:
                    if idx == idx_list[len(idx_list)//2+1]:
                        w_rand=0
                        h_rand=0
                    img = image_stack.data[idx,i+w_rand:i+patch_size+w_rand,
                                        j+h_rand:j+patch_size+h_rand]
                    img = np.array(img,dtype='float32')
                    gain = gain_value[i+w_rand:i+patch_size+w_rand,
                                    j+h_rand:j+patch_size+h_rand]
                    img = img*gain
                    if first:
                        first=False
                        img_series = np.expand_dims(img,0)
                    else:
                        img_series = np.concatenate((img_series,np.expand_dims(img,0)),0)
                img_series = np.concatenate((img_series,np.expand_dims(gain,0)),0)
                patches.append(img_series)
    index=0
    image_stack.close()
    for x in patches:
        if not os.path.exists(save_dir):
                os.mkdir(save_dir)
        np.savez_compressed(os.path.join(save_dir,
            f'set1_train_patch_{mrc_num}_{stack_num}_{index}.npz'), data=x)
        index=index+1

def generate_patch_memory_eficient_dm4(img_dir, gain_dir, save_dir, patch_size, stride, aug_times, processor_num=20, ratio=0.1, frame_num=5):
    patch_size, stride = patch_size, stride
    aug_times = aug_times
    save_dir = save_dir
    gain_dir = gain_dir
    src_dir = img_dir + '/'
    frames = frame_num
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(processes=processor_num)
    file_list = sorted(glob.glob(src_dir+'*.dm4'),) 
    input = []
    for file_num in range(0,len(file_list)):
        image_stack = dm.fileDM(file_list[file_num])
        stacks1, stacks2, width, height = image_stack.getMemmap(0).shape
        for s in range(0, stacks1*stacks2):
            idx_list = idxreturn(s, stacks1*stacks2, frames)
            input.append((file_list[file_num], gain_dir, idx_list, patch_size, stride, aug_times, save_dir, file_num, s, width, height, ratio))
    with tqdm(total=len(input)) as pbar:
        for _ in tqdm(pool.imap_unordered(map_function_dm4, input)):
            pbar.update()

    input = []
    pool.close()
    pool.join()

    return print('finish')

def map_function_dm4(i):
    return gen_patches_with_gainfix_dm4(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11])

def gen_patches_with_gainfix_dm4(file_name, gain_dir, idx_list, patch_size, stride, aug_times, save_dir, mrc_num, stack_num, width, height, ratio=0.1):
    image_stack = dm.fileDM(file_name)
    stacks1, stacks2, width, height = image_stack.getMemmap(0).shape
    if isinstance(gain_dir, type(None)):
        gain_value = np.ones((width, height))
    else:
        gain = dm.fileDM(gain_dir)
        gain_value = gain.getDataset(0).astype(np.float32)
        gain_value = np.flipud(gain_value)
    patches = []
    for i in range(0, width - patch_size + 1, stride):
        for j in range(0, height - patch_size + 1, stride):
            if random.random() < ratio:
                first = True
                w_rand = 0
                h_rand = 0
                for idx in idx_list:
                    if idx == idx_list[len(idx_list) // 2 + 1]:
                        w_rand = 0
                        h_rand = 0
                    s1 = idx//stacks2
                    s2 = idx%stacks2
                    img = image_stack.getMemmap(0)[s1,s2, i + w_rand:i + patch_size + w_rand,
                                                    j + h_rand:j + patch_size + h_rand].astype(np.float32)
                    gain = gain_value[i + w_rand:i + patch_size + w_rand,
                                     j + h_rand:j + patch_size + h_rand]
                    img = img * gain
                    if first:
                        first=False
                        img_series = np.expand_dims(img,0)
                    else:
                        img_series = np.concatenate((img_series,np.expand_dims(img,0)),0)
                img_series = np.concatenate((img_series,np.expand_dims(gain,0)),0)
                patches.append(img_series)
    index=0
    for x in patches:
        if not os.path.exists(save_dir):
                os.mkdir(save_dir)
        np.savez_compressed(os.path.join(save_dir,
            f'set1_train_patch_{mrc_num}_{stack_num}_{index}.npz'), data=x)
        index=index+1

def generate_patch_img(img_dir,gain_dir,save_dir,patch_size,stride,aug_times,frames=5,processor_num=20,ratio=0.1):
    patch_size, stride = patch_size, stride
    aug_times = aug_times
    save_dir = save_dir
    gain_dir = gain_dir
    src_dir = img_dir+'/'
    frames = frames
    src_name = os.path.basename(os.path.normpath(src_dir))
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(processes=processor_num)
    file_list = sorted(glob.glob(src_dir+'*.tif'))  
    input = []
    for file_num in range(0,len(file_list)):
        if file_num==0:
            img = cv.imread(file_list[file_num],flags=cv.IMREAD_UNCHANGED)
            #img = np.array(Image.open(file_list[file_num]),dtype='float32')
            width = img.data.shape[0]
            height = img.data.shape[1]
        idx_list = idxreturn(file_num,len(file_list),frames)
        input.append((file_list,gain_dir,idx_list,patch_size,stride,aug_times,save_dir,file_num,width,height,src_name,ratio))
    with tqdm(total=len(input)) as pbar:
        for _ in tqdm(pool.imap_unordered(map_function_img,input)):
            pbar.update()

    input =[]
    pool.close()
    pool.join()

    return print('finish')

def map_function_img(i):
    return gen_patches_with_gainfix_img(i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7],i[8],i[9],i[10],i[11])

def gen_patches_with_gainfix_img(file_list,gain_dir,idx_list,patch_size,stride,aug_times,save_dir,mrc_num,width,height,src_name,ratio):
    patches=[]
    if isinstance(gain_dir, type(None)):
        gain_value = np.ones((width,height))
    else:
        gain = mrcfile.mmap(gain_dir, mode='r')
        gain_value = gain.data 
        gain_value = np.flipud(gain_value)
    for i in range(0, width-patch_size+1, stride):
        for j in range(0, height-patch_size+1, stride):
            generator = np.random.choice([0,1], 1, p=[1-ratio,ratio])
            if generator[0]:
                first=True
                w_rand=random.choice([0])
                h_rand=random.choice([0])
                for idx in idx_list:
                    img = np.array(cv.imread(file_list[idx],flags=cv.IMREAD_UNCHANGED),dtype='float32')
                    #img = np.array(Image.open(file_list[idx]),dtype='float32')
                    img = img[i+w_rand:i+patch_size+w_rand,
                                        j+h_rand:j+patch_size+h_rand]
                    gain = gain_value
                    gain = gain[i+w_rand:i+patch_size+w_rand,
                        j+h_rand:j+patch_size+h_rand]
                    img = gain*img
                    if first:
                        first=False
                        img_series = np.expand_dims(img,0)
                    else:
                        img_series = np.concatenate((img_series,np.expand_dims(img,0)),0)
                for l in range(aug_times):
                    x_aug = data_aug(img_series,l)
                    patches.append(x_aug)
    index=0
    for x in patches:
        if not os.path.exists(save_dir):
                os.mkdir(save_dir)
        np.savez_compressed(os.path.join(save_dir,
            f'set1_train_patch_{mrc_num}_{index}_{src_name}.npz'), data=x)
        index=index+1

# --------------------------
# --- multiple dm4 files ---
# --------------------------
def generate_patch_dm4_frames(img_dir, gain_dir, save_dir, patch_size, stride, aug_times, frames=1, processor_num=20, ratio=0.1):
    patch_size, stride = patch_size, stride
    aug_times = aug_times
    save_dir = save_dir
    src_dir = os.path.join(img_dir, '')
    src_name = os.path.basename(os.path.normpath(src_dir))
    
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(processes=processor_num)
    
    file_list = sorted(glob.glob(src_dir + '*.dm4'))
    if not file_list:
        print(f"No .dm4 files found in {src_dir}")
        return

    input_args = []
    try:
        first_dm = dm.fileDM(file_list[0])
        temp_data = first_dm.getDataset(0)
        temp_img = temp_data['data'] if isinstance(temp_data, dict) else temp_data
        width, height = temp_img.shape
        del first_dm
    except Exception as e:
        print(f"Error reading dimensions from {file_list[0]}: {e}")
        return

    for file_num in range(len(file_list)):
        idx_list = idxreturn(file_num, len(file_list), frames)
        
        input_args.append((
            file_list, 
            gain_dir, 
            idx_list, 
            patch_size, 
            stride, 
            aug_times, 
            save_dir, 
            file_num, 
            width, 
            height, 
            src_name, 
            ratio
        ))

    print(f"Generating patches from {len(file_list)} DM4 files...")
    with tqdm(total=len(input_args)) as pbar:
        for _ in pool.imap_unordered(map_function_dm4_frames, input_args):
            pbar.update()

    pool.close()
    pool.join()
    print('finish')

def map_function_dm4_frames(args):
    return gen_patches_with_gainfix_dm4_frames(*args)

def gen_patches_with_gainfix_dm4_frames(file_list, gain_dir, idx_list, patch_size, stride, aug_times, save_dir, mrc_num, width, height, src_name, ratio):
    patches = []
    if gain_dir is None or isinstance(gain_dir, type(None)) or gain_dir == 'None':
        gain_value = np.ones((width, height), dtype=np.float32)
    else:
        if gain_dir.endswith('.dm4'):
            g_obj = dm.fileDM(gain_dir)
            g_data = g_obj.getDataset(0)
            gain_value = g_data['data'] if isinstance(g_data, dict) else g_data
        else:
            with mrcfile.open(gain_dir, permissive=True) as mrc:
                gain_value = mrc.data
        gain_value = np.flipud(gain_value).astype(np.float32)

    for i in range(0, width - patch_size + 1, stride):
        for j in range(0, height - patch_size + 1, stride):
            generator = np.random.choice([0, 1], 1, p=[1 - ratio, ratio])
            if generator[0]:
                first = True
                w_rand = random.choice([0]) 
                h_rand = random.choice([0])
                for idx in idx_list:
                    current_file = file_list[idx]
                    try:
                        dm_obj = dm.fileDM(current_file)
                        img_data = dm_obj.getDataset(0)
                        img = img_data['data'] if isinstance(img_data, dict) else img_data
                        img = img.astype(np.float32)
                    except:
                        img = np.zeros((width, height), dtype=np.float32)
                    
                    img_crop = img[i + w_rand : i + patch_size + w_rand,
                                   j + h_rand : j + patch_size + h_rand]
                    
                    gain_crop = gain_value[i + w_rand : i + patch_size + w_rand,
                                           j + h_rand : j + patch_size + h_rand]
                    
                    img_crop = img_crop * gain_crop
                    img_ready = np.expand_dims(img_crop, 0)
                    
                    if first:
                        first = False
                        img_series = img_ready
                    else:
                        img_series = np.concatenate((img_series, img_ready), axis=0)

                gain_ready = np.expand_dims(gain_crop, 0)
                #img_series = np.concatenate((img_series, gain_ready), axis=0)
                for l in range(aug_times):
                    x_aug = data_aug(img_series, l)
                    patches.append(x_aug)

    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError:
            pass

    index = 0
    for x in patches:
        filename = f'set1_train_patch_{mrc_num}_{index}_{src_name}.npz'
        np.savez_compressed(os.path.join(save_dir, filename), data=x)
        index += 1