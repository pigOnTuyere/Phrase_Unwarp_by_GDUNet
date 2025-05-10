import torch.nn.functional as f
import numpy as np
#  Contact qq 308128628 to obtain the complete code
#  Contact qq 308128628 to obtain the complete code

def gaussian_filter(images, sigma=1, kernel_size=5):
    """Apply Gaussian filter to a batch of images."""
    batch_size, C, H, W = images.shape
    device = images.device

    # Create a Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma).pow(2))
    g /= g.sum()
    kernel = g[:, None] * g[None, :]
    kernel = kernel.expand(C, 1, kernel_size, kernel_size)

    return f.conv2d(images, kernel, padding=kernel_size // 2, groups=C)

def sin_cos_n_step(I):

    pass

def calculate_wrap_phrase(I, mov=12):
    """

    :param I:
    :param mov:
    :return:
    """
    pass
def three_fre(phase, freq, device="cpu"):
    """
    三频外差法求取绝对相位 (支持 GPU 加速, 适用于 Batch 处理)

    :param phase: Tensor(batch_size, 3, H, W)，包裹相位
    :param freq: List[float]，三个频率
    :param device: "cpu" 或 "cuda"
    :return: (up, k1, phUnWrap, PH12, PH23, PH123) （freq[0]对应的包裹相位，k1）
    """
    ##
    pass

    # return up, k1, phUnWrap, PH12, PH23, PH123



def decode_muti_phase(Igs, freqs, threshold,mov=12):
    """
    多频外差法计算绝对相位，支持 batch 处理 (batch_size, C, H, W)

    :param Igs: Input images (batch_size, C, H, W)
    :param freqs: List of frequency  (3个值)
    :param threshold: 用于计算 B_mask 的阈值
    :return: (up_phase, B_avg, B_mask)
    """
    batch_size, C, H, W = Igs.shape
    device = Igs.device

    if isinstance(threshold, (int, float)):  # If threshold is a scalar
        threshold = torch.full((batch_size,), threshold, dtype=torch.float32, device=device)
    else:  # If the threshold is already a tensor
        threshold = torch.tensor(threshold, dtype=torch.float32).to(device)

    # Transform threshold to [batch size, 1, 1]
    threshold = threshold.view(batch_size, 1, 1)

    # Gaussian filter
    Igs = gaussian_filter(Igs)  # 假设

    # Calculate the wrapping phase
    ND, B, phase_tensor = calculate_wrap_phrase(Igs,mov)

    # 三频外差法求绝对相位
    up_phase, k, _, _, _, _ = three_fre(phase_tensor, freqs, device=device)

    # Calculate B1, B2, B3
    B1, B2, B3 = B[:, 0], B[:, 1], B[:, 2]  # 直接索引 batch 维度的通道

    # Calculate the B avg on the batch dimension
    B_avg = (B1 + B2 + B3) / 3  # 形状 (batch_size, H, W)

    B_mask = (B1 > threshold) & (B2 > threshold) & (B3 > threshold)  # 形状 (batch_size, H, W)

    # Check the validity of the label acquisition: When k > freqs[0], set its mask to 0
    k_mask = (0 <= k) & (k <= freqs[0])

    B_mask = B_mask & k_mask
    return up_phase, k, B_mask,ND





import os

def load_images_from_subfolders(data_folder, obj_dir_map, mov, postfix=".bmp", resize=None, device="cpu"):
    """
    读取多个子文件夹下的图片，并转换为 4D Tensor (batch_size, C, H, W)

    :param data_folder: str, 数据文件夹路径

    :param obj_dir_map: list[int], 需要处理的 map 数组
    :param mov: int, 运动参数
    :param postfix: str, 图片文件后缀，如 ".png"、".jpg"
    :param resize: tuple(int, int), 统一调整图片大小 (H, W)
    :param device: str, 运行设备 "cpu" or "cuda"
    :return: dict, {subfolder_name: Tensor(batch_size, C, H, W)}
    """

    # 生成 obj_dir 列表
    obj_dir = [0]
    for d in obj_dir_map:
        sequence = list(range((d - 1) * mov + 1, d * mov + 1))
        obj_dir.extend(sequence)

    transform_list = [transforms.ToTensor()]
    if resize:
        transform_list.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(transform_list)

    folder_tensors = []
    image_tensors = []

    for idx in obj_dir:
        filename = f"{idx:02d}{postfix}"  # 生成文件名 (补零格式)
        print(filename)
        file_path = os.path.join(data_folder, filename)

        if os.path.exists(file_path):
            image = Image.open(file_path).convert("L")
            image = np.array(image, dtype=np.float32)  # 直接转换为 float32
            image_tensor = transform(image).squeeze() # 转换为 Tensor
            image_tensors.append(image_tensor)

    if image_tensors:
        folder_tensors = torch.stack(image_tensors).to(device).unsqueeze(0)  # 合并为 4D Tensor
    tensor_repeated = folder_tensors.repeat(2, 1, 1, 1)  # 形状变为 [2, 3, 256, 256]
    return tensor_repeated


from scipy.io import savemat
def save_to_mat(up_phase, B_avg, B_mask,ND, save_path):
    """
    将 up_phase, B_avg, B_mask 保存为 .mat 文件

    :param up_phase: 解包裹相位 (batch_size, H, W)
    :param B_avg: 平均亮度 (batch_size, H, W)
    :param B_mask: 亮度掩码 (batch_size, H, W)
    :param save_path: 保存路径
    """
    # 将 PyTorch 张量转换为 NumPy 数组
    up_phase_np = up_phase.cpu().numpy()
    B_avg_np = B_avg.cpu().numpy()
    B_mask_np = B_mask.cpu().numpy()
    ND_np = ND.cpu().numpy()
    # 保存为 .mat 文件
    savemat(save_path, {
        "up_phase": up_phase_np,
        "B_avg": B_avg_np,
        "B_mask": B_mask_np,
        "ND":ND_np
    })


import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 测试代码
    # ----1 数据获取
    data_folder = r"K:\datasetsofConstruct\目标获取\cropped_images_9\113"
    obj_dir_map = [2, 3, 5]  # MATLAB 代码中的 map
    mov = 12
    freqs = [70, 11, 1]
    threshold = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folder_tensors = load_images_from_subfolders(data_folder, obj_dir_map, mov)

    # ----2 数据获取
    up_phase, B_avg, B_mask,ND = decode_muti_phase(folder_tensors, freqs, threshold)
    # ----3 保存结果
    save_path = "output_results.mat"  # 保存路径
    save_to_mat(up_phase, B_avg, B_mask, ND,save_path)
    print(f"Results saved to {save_path}")
