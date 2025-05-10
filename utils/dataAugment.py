import random
from torchvision.transforms import functional as F
import torch.nn.functional as nnf  # 确保使用 torch.nn.functional
import numpy as np
from utils.calculatePhrase import calculate_wrap_phrase, decode_muti_phase,sin_cos_n_step


def peaks(X, Y):
    """
    Generate a peaks function similar to MATLAB's peaks.
    :param X: X-coordinates (meshgrid array)
    :param Y: Y-coordinates (meshgrid array)
    :return: Z: Z-values computed using the peaks formula
    """
    Z = 3 * (1 - X) ** 2 * np.exp(-(X ** 2) - (Y + 1) ** 2) \
        - 10 * (X / 5 - X ** 3 - Y ** 5) * np.exp(-X ** 2 - Y ** 2) \
        - 1 / 3 * np.exp(-(X + 1) ** 2 - Y ** 2)
    return Z


class SynchronizedTransform:
    def __init__(self, transforms, mode=2):
        '''
        Accept a list of a series of transformation operations to ensure that the images and labels are transformed synchronously
        :param transforms: List of transformations (each must accept both image and label)
        :param mode: mode of data augment
        '''
        self.transforms = transforms
        self.mode = mode

    def __call__(self, img, label=None):

        if self.mode == 1:
            '''
            Mode 1: Input both the image and the label simultaneously to ensure that the image and the label undergo synchronous transformation. 
            This mode can be used when predicting ND. 
            The enhancement mode can be adopted as RandomRotation, RandomShift and movement
            '''
            # Randomly select a transform
            transform = random.choice(self.transforms)
            img, label = transform(img, label)


            return img, label

        if self.mode == 2:
            '''
            Mode 2: The label is obtained by calculating the fringe image
            '''
            transform = random.choice(self.transforms)
            if isinstance(transform, (list, tuple)):
                img_transformed = img
                for t in transform:
                    img_transformed, _ = t(img_transformed)
            else:
                img_transformed, _ = transform(img)
            # 多频外差法求条纹级次k，掩膜mask
            _, k, B_mask, ND = decode_muti_phase(Igs=img_transformed.unsqueeze(0), freqs=[70,12, 11], threshold=50,
                                                 mov=12)
            k = k * B_mask
            ND = ND * B_mask

            # channels_to_extract = [1]  # Extract the first image of the first frequency
            channels_to_extract = [1, 13, 25]  # Extract the first image of each frequency
            extracted_data = img_transformed[channels_to_extract, :, :]

            return extracted_data.squeeze(0), ND.squeeze(0),k.squeeze(0)


class RandomZoom:
    '''
    Random zoom data augmentation.
    - Zoom in: Randomly crop the image after scaling up.
    - Zoom out: Pad the image after scaling down.
    '''

    def __init__(self, scale_range=(0.8, 1.2), p=1):
        '''
        Args:
            scale_range (tuple): Min and max scale ratio (e.g., 0.8 = 80% size).
            p (float): Probability of applying augmentation.
        '''
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img, label=None):
        if random.random() < self.p:
            # Generate random scale factor
            scale = random.uniform(*self.scale_range)
            h, w = img.shape[-2:]  # Assume img is [C, H, W]

            # Calculate new dimensions
            new_h = int(h * scale)
            new_w = int(w * scale)

            # Resize the image
            img = nnf.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)

            # Handle labels if provided (same interpolation)
            if label is not None:
                label = nnf.interpolate(label.unsqueeze(0).float(), size=(new_h, new_w), mode='nearest').squeeze(0).long()

            # Case 1: Zoomed out (scale < 1) -> Pad to original size
            if scale < 1:
                pad_h = h - new_h
                pad_w = w - new_w
                # Pad symmetrically on both sides
                img = nnf.pad(img, (pad_w // 2, pad_w - pad_w // 2,
                                  pad_h // 2, pad_h - pad_h // 2),
                            mode='constant', value=0)
                if label is not None:
                    label = nnf.pad(label, (pad_w // 2, pad_w - pad_w // 2,
                                          pad_h // 2, pad_h - pad_h // 2),
                                  mode='constant', value=0)

            # Case 2: Zoomed in (scale > 1) -> Crop to original size
            elif scale > 1:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                img = img[:, start_h:start_h + h, start_w:start_w + w]
                if label is not None:
                    label = label[:, start_h:start_h + h, start_w:start_w + w]

        return img, label




class NoTransform:
    def __call__(self, img, label=None):
        """
        Return the input directly without any transformation.
        """
        return img, label


class RandomRotation:
    '''
        Random rotation data enhancement
    '''
    def __init__(self, degrees=10, p=1):
        self.degrees = degrees
        self.p = p

    def __call__(self, img, label=None):
        if random.random() < self.p:
            angle = random.randint(-self.degrees, self.degrees)
            img = F.rotate(img, angle)
            if label is not None:
                label = F.rotate(label, angle)
        return img, label



class RandomShift:
    def __init__(self, max_shift=100, p=1):
        """
        Random translation data augmentation
        :param max_shift:Maximum number of translational pixels (the range of absolute values in the horizontal and vertical directions)
        :param p: Trigger probability
        """
        self.max_shift = max_shift
        self.p = p

    def __call__(self, img, label=None):
        if random.random() < self.p:

            shift_x = random.randint(-self.max_shift, self.max_shift)
            shift_y = random.randint(-self.max_shift, self.max_shift)


            img = F.affine(img, angle=0, translate=(shift_x, shift_y), scale=1, shear=0, fill=0)
            if label is not None:
                label = F.affine(label, angle=0, translate=(shift_x, shift_y), scale=1, shear=0, fill=0)
        return img, label



class RandomWrap:
    def __init__(self, p=1, warp_type='sin'):
        """
        Random distortion data augmentation
        :param p: The probability of applying the transformation
        :param warp_type: Select the type of distortion, which can be 'sin' or 'peaks'
        """
        self.p = p
        self.warp_type = warp_type

    def __call__(self, img, label=None):
        """
        Apply the transformation to images and labels.
        """
        if random.random() < self.p:
            if self.warp_type == 'sin':
                img, label = self.sin_wrap(img, label)
            elif self.warp_type == 'peaks':
                img, label = self.peaks_wrap(img, label)
        return img, label

    def _generate_distortion_coords(self, img, distortion_func):
        """
        Generate the distorted coordinates and apply the given distortion function.
        :param img: Input the image tensor
        :param distortion_func: Distortion function
        :return: New image coordinates
        """
        C, rows, cols = img.shape
        # 创建网格坐标
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))

        # 应用扭曲函数
        x_new, y_new = distortion_func(x, y)

        # 将目标坐标归一化为 [-1, 1] 区间
        x_new = 2 * x_new / (cols - 1) - 1
        y_new = 2 * y_new / (rows - 1) - 1

        # 堆叠为 [H, W, 2] 的格式
        coords = np.stack((x_new, y_new), axis=-1)

        # 生成网格坐标 (N, H, W, 2) 格式
        grid = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)

        return grid

    def _apply_warp(self, img, label, grid):
        """
        使用 grid_sample 应用空间变换。
        :param img: 输入图像
        :param label: 输入标签
        :param grid: 变换后的网格
        :return: 扭曲后的图像和标签
        """
        img_warped = nnf.grid_sample(img.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        if label is not None:
            label_warped = nnf.grid_sample(label.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros',
                                           align_corners=True)
            return img_warped.squeeze(0), label_warped.squeeze(0)
        else:
            return img_warped.squeeze(0), None

    def sin_wrap(self, img, label=None, wave_amplitude=2, wave_frequency=50):
        """
        使用正弦波形对图像进行波形扭曲变换。
        """
        wave_amplitude = random.uniform(0.5, wave_amplitude)

        def distortion_func(x, y):
            x_new = x + wave_amplitude * np.sin(2 * np.pi * y / wave_frequency)  # 水平方向波形扭曲
            y_new = y
            return x_new, y_new

        grid = self._generate_distortion_coords(img, distortion_func)
        return self._apply_warp(img, label, grid)

    def peaks_wrap(self, img, label=None):
        """
        使用 peaks 函数生成的扭曲场对图像进行变换。
        """

        def distortion_func(x, y):
            # 将 x 和 y 标准化为 [-3, 3] 区间，与 MATLAB 的 peaks 函数类似
            x_norm = 6 * (x / x.shape[1] - 0.5)
            y_norm = 6 * (y / y.shape[0] - 0.5)

            # 调用 peaks 函数
            Z_peaks = peaks(x_norm, y_norm)

            # 缩放以适应图像扭曲场
            Z_peaks_scaled = random.uniform(1, 4) * Z_peaks  # 随机缩放因子
            x_new = x + Z_peaks_scaled  # 扭曲 X 坐标
            y_new = y
            return x_new, y_new

        grid = self._generate_distortion_coords(img, distortion_func)
        return self._apply_warp(img, label, grid)


class RandomBrightness:
    def __init__(self, max_delta=0.2, p=0.0):
        """
        随机亮度增强类

        参数：
        - max_delta: 亮度调整的最大幅度，范围在 [0, 1] 之间
        - p: 执行亮度增强的概率
        """
        self.max_delta = max_delta
        self.p = p

    def __call__(self, img, label=None):
        if random.random() < self.p:
            # 随机生成亮度调整幅度
            delta = random.uniform(-0.3, self.max_delta)

            # 对图像进行亮度调整
            img = F.adjust_brightness(img, 1 + delta)

            # 如果提供了标签，可以选择是否对标签进行相同的亮度调整
            if label is not None:
                label = F.adjust_brightness(label, 1 + delta)

        return img, label


def show_images(img, label, img_aug, label_aug):
    # 将张量转为可显示的 numpy 格式
    img = transforms.ToPILImage()(img)
    label = transforms.ToPILImage()(label)
    img_aug = transforms.ToPILImage()(img_aug)
    label_aug = transforms.ToPILImage()(label_aug)
    plt.imsave("augment.jpg", label_aug, cmap="gray")

    # 绘图
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(img,cmap="gray")
    axs[0, 0].set_title("Original Image")
    axs[0, 1].imshow(label)
    axs[0, 1].set_title("Original Label")

    axs[1, 0].imshow(img_aug,cmap="gray")
    axs[1, 0].set_title("Augmented Image")
    axs[1, 1].imshow(label_aug)
    axs[1, 1].set_title("Augmented Label")

    for ax in axs.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# import os
# import cv2
# # 加载图片 调试sin_cos_n_step
# def load_images_to_tensor(folder_path):
#     """
#     读取指定文件夹中的所有图像并转为 PyTorch 张量。

#     参数:
#         folder_path (str): 文件夹路径。

#     返回:
#         torch.Tensor: 形状为 [B, C, H, W] 的 4D 张量。
#     """
#     image_list = []
#     for filename in sorted(os.listdir(folder_path)):  # 确保按照文件名排序
#         if filename.endswith((".png", ".bmp", ".jpg")):
#             img_path = os.path.join(folder_path, filename)
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
#             image_list.append(img)

#     # 将图像列表转为 numpy 数组，再转为 torch 张量
#     images = np.stack(image_list, axis=0)  # 形状 [B, H, W]
#     tensor_images = torch.from_numpy(images).float() # 归一化到 [0, 1]
#     tensor_images = tensor_images.unsqueeze(0)  # 添加通道维度，形状 [B, 1, H, W]
#     return tensor_images


# class RandomCrop:
#     def __init__(self, crop_size):
#         self.crop_size = crop_size
#
#     def __call__(self, img, label):
#         i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.crop_size)
#         img = F.crop(img, i, j, h, w)
#         label = F.crop(label, i, j, h, w)
#         return img, label


import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 2 测试数据加强效果
    import scipy.io

    imgs_path = r'E:\python_project\Phrase_Unwarp_by_GDUNet\data\32_imgs.mat'
    try:
        imgs = scipy.io.loadmat(imgs_path)["images"]
    except KeyError:
        raise ValueError(f"'images' 键在 {imgs_path} 中不存在。")

    imgs = np.array(imgs, dtype=np.float32)
    imgs_tensor = torch.from_numpy(imgs)


    synchronized_transform = SynchronizedTransform([
        RandomZoom()
    ])

    inputTensor, ND, k = synchronized_transform(imgs_tensor)
    # 应用数据增强
    # # 生成一个假设的标签，形状和图像一致（单通道）
    # label = (torch.rand(1, img.shape[1], img.shape[2]) > 0.5).float()  # 二值分割标签

    #  显示结果
    show_images(imgs_tensor[1,:,:]/255, ND[0,:,:], inputTensor[0,:,:]/255, k)

    # 3. 调试高斯滤波
    # 读取图像
    # import cv2
    # img = cv2.imread(r'D:\code\deeplearning+reconstruction\Phase_unwrapping_by_U-Net-main\test.png', cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
    #
    # # 检查图像是否加载成功
    # if img is None:
    #     print("Image not loaded correctly")
    #     exit()
    #
    # # 将图像转换为Tensor，形状为 [1, H, W]（加上batch维度）
    # img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # 转换为 [1, H, W]
    #
    # # 通过高斯滤波器应用滤波
    # filtered_img_tensor = apply_gaussian_filter(img_tensor, kernel_size=5, sigma=2.0)
    #
    # # 转换回numpy数组以便显示
    # filtered_img = filtered_img_tensor[0].numpy()
    #
    # # 显示原始图像和滤波后的图像
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(img, cmap='gray')
    # plt.imsave("orign.jpg",img)
    # plt.title("Original Image")
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(filtered_img, cmap='gray')
    # plt.imsave("filtered_img.jpg", filtered_img)
    # plt.title("Filtered Image")
    # plt.axis('off')
    #
    # plt.show()


