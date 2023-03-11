import numpy as np
import numbers
import random
import cv2
from scipy.ndimage import gaussian_filter
import torch
import torchvision

class GaussianBlurVideo(object):
    def __init__(
        self, sigma_min=[0.0, 0.1], sigma_max=[0.0, 2.0], use_PIL=False
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, frames):
        sigma_y = sigma_x = random.uniform(self.sigma_min[1], self.sigma_max[1])
        sigma_t = random.uniform(self.sigma_min[0], self.sigma_max[0])
        frames = gaussian_filter(frames, sigma=(0.0, sigma_t, sigma_y, sigma_x))
        # frames = torch.from_numpy(frames)
        return frames
    

class RandomGaussianBlurVideo(object):
    def __init__(
        self, sigma_min=[0.0, 0.1], sigma_max=[0.0, 2.0], use_PIL=False,p = 0.3
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p

    def __call__(self, frames):
        if random.random() < self.p:
            sigma_y = sigma_x = random.uniform(self.sigma_min[1], self.sigma_max[1])
            sigma_t = random.uniform(self.sigma_min[0], self.sigma_max[0])
            frames = gaussian_filter(frames, sigma=(0.0, sigma_t, sigma_y, sigma_x))
        # frames = torch.from_numpy(frames)
        return frames
    
class ToTensor(object):
    def __init__(self):
        pass
    
    def __call__(self, frames):
        frames = torch.from_numpy(frames)
        frames = frames.permute(3, 0, 1, 2)
        
        return frames
    
class Random_Resized_crop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)):
        self.raondom_resized_crop = torchvision.transforms.RandomResizedCrop(size, scale=(0.4, 0.5), ratio = ratio)
        
    def __call__(self, frames):
        frames = self.raondom_resized_crop(frames)
        
        return frames


class RGB(object):
    """Crop the given video sequences (t x h x w) at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self):
        pass

    def __call__(self, imgs):
        imgs = imgs[:, :, :, [2, 1, 0]]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class RandomCrop(object):
    """Crop the given video sequences (t x h x w) at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        t, h, w, c = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h!=th else 0
        j = random.randint(0, w - tw) if w!=tw else 0
        return i, j, th, tw

    def __call__(self, imgs):
        
        i, j, h, w = self.get_params(imgs, self.size)

        imgs = imgs[:, i:i+h, j:j+w, :]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class ReSize(object):
    """Crop the given video sequences (t x h x w) at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        imgs = [cv2.resize(img, dsize=self.size) for img in imgs]
        imgs = np.concatenate(
            [np.expand_dims(img, axis=0) for img in imgs],
            axis=0,
        )

        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class CenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i+th, j:j+tw, :]


    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomHorizontalFlip(object):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        if random.random() < self.p:
            # t x h x w
            return np.flip(imgs, axis=2).copy()
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

import math

def _get_param_spatial_crop(
    scale, ratio, height, width, num_repeat=10, log_scale=True, switch_hw=False
):
    """
    Given scale, ratio, height and width, return sampled coordinates of the videos.
    """
    for _ in range(num_repeat):
        area = height * width
        target_area = random.uniform(*scale) * area
        if log_scale:
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
        else:
            aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if np.random.uniform() < 0.5 and switch_hw:
            w, h = h, w

        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


class Random_Resized_Crop_with_Shift(object):

    def __init__(self, size, scale = (0.8, 1.0), ratio=(3.0 /4.0, 4.0 /3.0)):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        
        self.scale = scale
        self.ratio = ratio
        
    def __call__(self,
        images
    ):
        """
        This is similar to random_resized_crop. However, it samples two different
        boxes (for cropping) for the first and last frame. It then linearly
        interpolates the two boxes for other frames.
        Args:
            images: Images to perform resizing and cropping.
            target_height: Desired height after cropping.
            target_width: Desired width after cropping.
            scale: Scale range of Inception-style area based random resizing.
            ratio: Aspect ratio range of Inception-style area based random resizing.
        """
        t = images.shape[1]
        height = images.shape[2]
        width = images.shape[3]

        target_height = self.size[0]
        target_width= self.size[1]
        
        scale = self.scale
        ratio = self.ratio

        j, i, h, w = _get_param_spatial_crop(scale, ratio, height, width)
        j_, i_, h_, w_ = _get_param_spatial_crop(scale, ratio, height, width)
        i_s = [int(i) for i in torch.linspace(i, i_, steps=t).tolist()]
        j_s = [int(i) for i in torch.linspace(j, j_, steps=t).tolist()]
        h_s = [int(i) for i in torch.linspace(h, h_, steps=t).tolist()]
        w_s = [int(i) for i in torch.linspace(w, w_, steps=t).tolist()]
        out = torch.zeros((t, target_height, target_width, 3))
        images = torch.from_numpy(images)
        print(images)
        for ind in range(t):
            out[ind : ind + 1, :, :, :] = torch.nn.functional.interpolate(
                images[
                    ind : ind + 1,
                    i_s[ind] : i_s[ind] + h_s[ind],
                    j_s[ind] : j_s[ind] + w_s[ind],
                    :,
                ],
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
        return out.numpy()

