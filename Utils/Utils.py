import torch
import numpy as np
import torch.nn as nn
import numba

def subset_sampler(idx,stack_num,frame_num=5):
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

@torch.jit.script
def torch_zscore_normalize(image):
    if image.dim() == 2:
        # Compute mean and std over the entire image for 2D images
        mean = image.mean()
        std = image.std()
        normalized_image = (image - mean) / std
    elif image.dim() == 3:
        # Compute mean and std over each channel for 3D images
        mean = image.mean(dim=(-2,-1), keepdim=True)
        std = image.std(dim=(-2,-1), keepdim=True)
        normalized_image = (image - mean) / std
    elif image.dim() == 4:
        # Compute mean and std over each channel for each image in the batch for 4D images
        mean = image.mean(dim=(-2,-1), keepdim=True)
        std = image.std(dim=(-2,-1), keepdim=True)
        normalized_image = (image - mean) / std
    else:
        raise ValueError("Unsupported image dimension. Image must be 2D, 3D, or 4D.")

    return normalized_image

def numpy_normalize(image):
    if len(image.shape)==2:
        v = image[:]
        image =(image-v.min())/(v.max()-v.min())
    if len(image.shape)==3:
        channels = image.shape[0]
        for c in range(channels):
            v = image[c,:]
            image[c,:,:] =(image[c,:,:]-v.min())/(v.max()-v.min())
    if len(image.shape)==4:
        batches = image.shape[0]
        channels = image.shape[1]
        for b in range(batches):
            for c in range(channels):
                v = image[b,c,:]
                image[b,c,:,:] =(image[b,c,:,:]-v.min())/(v.max()-v.min())
    return image

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
def numpy_meanzero(image):
    output = np.empty_like(image)
    channels = image.shape[0]
    for c in range(channels):
        v = image[c,:,:].flatten()
        mean = v.mean()
        std = v.std()
        output[c,:,:] = (image[c,:,:] - mean)
    return output

def numpy_zscore_normalize_test(image):
    if len(image.shape)==2:
        image_nonscale=image.clone()
        v = image.flatten()
        image =(image-v.mean())/(v.std())
    if len(image.shape)==3:
        channels = image.shape[0]
        image_nonscale=image[channels//2,:,:].clone()
        for c in range(channels):
            v = image[c,:,:].flatten()
            image[c,:,:] =(image[c,:,:]-v.mean())/(v.std())
    if len(image.shape)==4:
        batches = image.shape[0]
        channels = image.shape[1]
        image_nonscale=image[:,channels//2,:,:].clone()
        for b in range(batches):
            for c in range(channels):
                v = image[b,c,:,:].flatten()
                mean=v.mean()
                std=v.std()
                image[b,c,:,:] =(image[b,c,:,:]-mean)/(std)
    return image, image_nonscale

def numpy_zscore_recover(image,original):
    image_copy = image.clone()
    if len(image.shape)==2:
        v = original.flatten()
        min=v.min()
        max=v.max()
        v2 = image[:,:].flatten()
        v2min=v2.min()
        v2max=v2.max()
        image_copy[:,:] =(image[:,:]-v2min)/(v2max-v2min)*(max-min)+min
    if len(image.shape)==3:
        channels = image.shape[0]
        for c in range(channels):
                v = original.flatten()
                min=v.min()
                max=v.max()
                v2 = image[c,:,:].flatten()
                v2min=v2.min()
                v2max=v2.max()
                image_copy[c,:,:] =(image[c,:,:]-v2min)/(v2max-v2min)*(max-min)+min
    if len(image.shape)==4:
        batches = image.shape[0]
        channels = image.shape[1]
        for b in range(batches):
            for c in range(channels):
                v = original.flatten()
                min=v.min()
                max=v.max()
                v2 = image[b,c,:,:].flatten()
                v2min=v2.min()
                v2max=v2.max()
                image_copy[b,c,:,:] =(image[b,c,:,:]-v2min)/(v2max-v2min)*(max-min)+min
    return image_copy

@numba.jit(nopython=True, cache=True)
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

class MixedLoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0):
        super().__init__()
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self,img1,img2):
        loss_a = self.mae(img1,img2)
        loss_s = self.mse(img1,img2)
        return loss_a + loss_s

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft

class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight