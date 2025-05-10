import os
import numpy as np
import scipy.io as sio
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
from utils.dataAugment import *
import random
from PIL import Image
import scipy.io


# 1
# 1. 输入数据条纹图像，标签是mat数据，

class PhaseDataset(Dataset):
    def __init__(self, img_dir, dir_gt, transform, extension='.mat'):
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.dir_gt = dir_gt
        self.extension = extension
        self.transform = transform

    ' Ask for input and ground truth'

    def __getitem__(self, index):
        # Get an ID of the input and ground truth
        img_path = os.path.join(self.img_dir, self.img_list[index])
        image = Image.open(img_path)
        image = transforms.ToTensor()(image)
        #
        base_name = self.img_list[index].rsplit(".", 1)[0]
        gt = os.path.join(self.dir_gt, base_name + "_3ND" + self.extension)

        ND = sio.loadmat(gt)["ND"]
        # Open them
        # 12 步相位移动时要除以3
        gt = transforms.ToTensor()(ND) / 3

        if self.transform:
            image, gt = self.transform(image, gt)
        return image.float(), gt.float(), 0

    ' Length of the dataset '

    def __len__(self):
        return len(self.img_list)


def load_data(**kwargs):
    '''

    :param kwargs:
    :return:
    '''
    '''
    This input mode (requires the input images- the composite image of the three frequency projection stripes and 
    labels- the numerators and denominators of the three frequency wrapping phases to be prepared in advance)
    Only RandomShift(),RandomRotation(), and NoTransform() can be used for data augmentation below. 
    Other data augmentation types are not allowed
    
    '''
    root_path = kwargs["data"]
    image_train =os.path.join(root_path,r"train\images")
    gt_train = os.path.join(root_path,r"train\labels")
    image_val = os.path.join(root_path,r"val\images")
    gt_val = os.path.join(root_path,r"val\labels")
    batch_size = kwargs["batch_size"]

    train_transform = SynchronizedTransform([
        RandomShift(),
        RandomRotation(),
        NoTransform()
    ],mode=1)

    val_transform = SynchronizedTransform([
        NoTransform()
    ])
    train_dataset = PhaseDataset(image_train, gt_train, train_transform)

    val_dataset = PhaseDataset(image_val, gt_val, val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader



