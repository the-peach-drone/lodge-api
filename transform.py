import cv2
import numpy as np
import random
import torch
import torchvision.transforms as transforms

# TODO: SPLIT TRANSFORM AS FUNCTIONS
def preprocessing(img, mask):
    img, mask = split_image(img, mask)

    vi = get_vi(img, "ExGR")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img[:, :, 3] = vi

    img = cv2.resize(img, dsize=(int(512), int(512)), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, dsize=(int(512), int(512)), interpolation=cv2.INTER_AREA)
    
    # img, mask = random_crop(img, mask, 512)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img, mask = img/255., mask/255.
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    return image_transform(img).float(), image_transform(mask).float()
    
def random_crop(img, mask, crop_size):
    h, w, c = img.shape
    h_r = random.randint(0, h - (crop_size))
    w_r = random.randint(0, w - (crop_size))
    img = img[h_r:h_r+crop_size, w_r:w_r+crop_size, :]
    mask = mask[h_r:h_r+crop_size, w_r:w_r+crop_size]
    return img, mask

def split_image(img, mask):
    h, w, c = img.shape
    img_crop = [
        img[:int(int(h/2)), :int(w/2), :],
        img[int(h/2):, :int(w/2), :],
        img[:int(h/2), int(w/2):, :],
        img[int(h/2):, int(w/2):, :]
    ]
    msk_crop = [
        mask[:int(h/2), :int(w/2), :],
        mask[int(h/2):, :int(w/2), :],
        mask[:int(h/2), int(w/2):, :],
        mask[int(h/2):, int(w/2):, :]
    ]
    sel = random.randint(0, 3)
    return img_crop[sel], msk_crop[sel]

def get_vi(img, vi=None):
    if vi == "ExG":
        return 2 * img[:, :, 1] - img[:, :, 0] - img[:, :, 2]
    elif vi == "ExR":
        return 1.4 * img[:, :, 0] - img[:, :, 1]
    elif vi == "ExGR":
        return 3 * img[:, :, 1] - 2.4 * img[:, :, 2] - img[:, :, 0]
    
def preprocessing_1(img, mask):
    img, mask = split_image(img, mask)

    img = cv2.resize(img, dsize=(int(512), int(512)), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, dsize=(int(512), int(512)), interpolation=cv2.INTER_AREA)
    
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img, mask = img/255., mask/255.
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    return image_transform(img).float(), image_transform(mask).float()
    
def preprocessing_2(img, mask):
    img, mask = random_crop(img, mask, 512)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img, mask = img/255., mask/255.
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    return image_transform(img).float(), image_transform(mask).float()
    
def preprocessing_1_4(img, mask):
    img, mask = split_image(img, mask)

    vi = get_vi(img, "ExGR")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img[:, :, 3] = vi

    img = cv2.resize(img, dsize=(int(512), int(512)), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, dsize=(int(512), int(512)), interpolation=cv2.INTER_AREA)
    
    # img, mask = random_crop(img, mask, 512)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img, mask = img/255., mask/255.
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    return image_transform(img).float(), image_transform(mask).float()
    