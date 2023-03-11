"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


# Tensor GN pipeline
import torch
def gauss_noise_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)
    
    sigma = 0.25
    
    out = img + sigma * torch.randn_like(img)
    
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return out


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True, src=False):
    transform_list = []

    # raise Exception
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))
    elif 'scale_unet' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale__unet(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
        
    if src:
        transform_list.append(transforms.Lambda(lambda img: __augment(img)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))    

    transform_list.append(transforms.Lambda(lambda img: __convert_3_channels(img)))


    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]


    if False:
        transform_list.append(transforms.RandomErasing())
        # transform_list.append(gauss_noise_tensor)
        

    return transforms.Compose(transform_list)

def __convert_3_channels(img):
    # print(img.shape)
    # x3d = np.repeat(np.expand_dims(img, axis=3), 3, axis=3)
    return img.convert('RGB')

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)

def __frame_image_cv2(pil_img, size):
    open_cv_image = np.array(pil_img)
    # Convert RGB to BGR
    img = open_cv_image[:, :, ::-1].copy()

    h = img.shape[0]
    w = img.shape[1]

    # Frame our target image 
    back = np.ones(size, dtype=np.uint8)*235
    hh, ww, _ = back.shape
    # print(f'hh, ww = {hh}, {ww}')

    # compute xoff and yoff for placement of upper left corner of resized image
    yoff = round((hh-h)/2)
    xoff = round((ww-w)/2)
    # print(f'xoff, yoff = {xoff}, {yoff}')

    # use numpy indexing to place the resized image in the center of background image
    result = back.copy()
    result[yoff:yoff+h, xoff:xoff+w] = img

    # convert back to PIL
    img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    return im_pil

def __scale__unet(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    import math
    print(f'{ow}, {oh}')
    net_size = 256
    wr = math.ceil (ow / net_size)
    hr = math.ceil (oh / net_size)
    print(f'{wr}, {hr}')
    ow = wr * net_size
    oh = hr * net_size
    # if ow == target_size :
    #     return img
    w = ow
    h = oh
    return  __frame_image_cv2(img,(h, w, 3))

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img



def __augment(pil_img):
    import random
    import string
    """Augment imag and mask"""
    import imgaug as ia
    import imgaug.augmenters as iaa
    import imgaug.parameters as iap
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
 
    open_cv_image = np.array(pil_img)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    # random number in range
    from random import randint
    import random

    seq = iaa.Sequential([
        
        # sometimes(iaa.JpegCompression(compression=(70, 99))),
        # iaa.ReplaceElementwise(0.05, [0, 255], per_channel=0.5)
        iaa.SaltAndPepper(0.001, per_channel=False),
        # iaa.ReplaceElementwise(0.01, iap.Normal(128, 0.4*128), per_channel=False)
        
        # sometimes(
        #     iaa.BlendAlphaElementwise(
        #         (0.0, random.uniform(0.2, 1.0)),
        #         foreground=iaa.Add(randint(60, 120)),
        #         background=iaa.Multiply(random.uniform(0.01, 0.4)))
        # ),

        # sometimes(iaa.ReplaceElementwise(0.01, iap.Normal(randint(100, 140), 0.4*randint(100, 140)), per_channel=False)),
        # sometimes(iaa.JpegCompression(compression=(90, 99)))
        
        # sometimes(iaa.OneOf([
        #     # iaa.GaussianBlur((0, 2.0)),
        #     # iaa.AverageBlur(k=(2, 7)),
        #     iaa.MedianBlur(k=(1, 3)),
        # ])),

        # sometimes(
        #     iaa.ElasticTransformation(alpha=(0.5, .8), sigma=0.25)
        # ),

    ], random_order=True)

    image_aug = seq(image=open_cv_image)

    # convert back to PIL
    img = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    return im_pil

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


