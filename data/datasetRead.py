import os
import numpy as np
import scipy.io as sio
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
from utils.dataAugment import *
import random
import scipy.io
#  Contact qq 308128628 to obtain the complete code

# 1
# 1. 输入数据条纹图像，
class PhaseDataset(Dataset):
    def __init__(self, data_path, sample_ratio, mov=12, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.mov = mov
        self.seed = 50

        #  # Obtain the paths of all the.mat files
        self.img_list = [f for f in os.listdir(data_path) if f.endswith('.mat')]
        self.sample_ratio = sample_ratio
        # Randomly sample some data
        if self.sample_ratio < 1:
            random.seed(self.seed)
            num_samples = int(len(self.img_list) * self.sample_ratio)
            self.img_list = random.sample(self.img_list, num_samples)

    def __getitem__(self, index):
        imgs_path = os.path.join(self.data_path, self.img_list[index])

        try:
            imgs = scipy.io.loadmat(imgs_path)["images"]
        except KeyError:
            raise ValueError(f"'images' 键在 {imgs_path} 中不存在。")

        imgs = np.array(imgs, dtype=np.float32)
        imgs_tensor = torch.from_numpy(imgs)


        if self.transform:
            # 3频外参法，12步相位移，因此mat数据包含37张图片（12*3+1，1是空白投影图片）
            assert imgs_tensor.shape[0] == 37, f"{imgs_path}的数据错误"

            input_tensor, ND, k = self.transform(imgs_tensor)
            input_tensor = input_tensor /255  # 输入图像像素归一
            ND = ND/3.0  # 12步时标签除以3更容易收敛
            return input_tensor, ND, k.long()
        else:
            return imgs_tensor

    def __len__(self):
        """返回数据集的样本数量。"""
        return len(self.img_list)






def load_data(**kwargs):
    '''

    :param kwargs:
    :return:
    '''
    root_path = kwargs["data"]
    train_path = os.path.join(root_path,"train")
    val_path = os.path.join(root_path,"val")
    batch_size =  kwargs["batch_size"]
    '''
    # Any data augmentation mode that can be carried out and provided under this mode 
    # (the original data is 37 pictures, and then the input images and labels are calculated)
    '''

    train_transform = SynchronizedTransform([
        # [RandomRotation(degrees=10), RandomShift(max_shift=200)],
        # [RandomShift(max_shift=200),
        # RandomWrap(warp_type='peaks')],
        # [RandomWrap(warp_type='sin'),
        # RandomBrightness()],RandomBrightness(),
        NoTransform()
    ],mode=2)
    val_transform = SynchronizedTransform([
        NoTransform()
    ])
    train_dataset = PhaseDataset(train_path, sample_ratio=1.0, transform=train_transform)
    val_dataset = PhaseDataset(val_path, sample_ratio=1.0, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    return train_loader, val_loader


from scipy.io import savemat





