import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):

        mse_loss = self.mse(y_pred, y_true)
        
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss


class Wrap_RMSELoss(nn.Module):
    def __init__(self):
        super(Wrap_RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # 
        wrap_pre = - torch.atan2(y_pred[:, 0, :, :], y_pred[:, 1, :, :])
        wrap_gt = - torch.atan2(y_true[:, 0, :, :], y_true[:, 1, :, :])
        mse_loss = self.mse(wrap_pre, wrap_gt)
        rmse_loss = torch.sqrt(mse_loss)

        return rmse_loss



class MixLoss1(nn.Module):
    def __init__(self):
        super(MixLoss1, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, y_pred, y_true):
        # 计算 MSE
        mse_loss = self.mse(y_pred, y_true)
        l1_loss = self.l1(y_pred, y_true)
        loss = mse_loss * 0.5 + 0.5 * l1_loss
        return loss


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, channels=2):
        """
        SSIM Loss
        :param window_size: Sliding window size
        :param sigma: The standard deviation of the Gaussian kernel
        :param channels: The number of channels of the image
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channels = channels
        self.gaussian_kernel = self._create_gaussian_kernel(window_size, sigma)

    def _create_gaussian_kernel(self, window_size, sigma):
        """
        create Gaussian kernel
        """
        kernel = torch.tensor([self._gaussian(x, sigma) for x in range(-window_size // 2 + 1, window_size // 2 + 1)])
        kernel = kernel.unsqueeze(1) * kernel.unsqueeze(0)  #
        kernel = kernel / kernel.sum()  #
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # expand to (1, 1, window_size, window_size)
        kernel = kernel.expand(self.channels, 1, window_size,
                               window_size)  # expand to (channels, 1, window_size, window_size)
        return kernel

    def _gaussian(self, x, sigma):
        """
        Calculate the value of the Gaussian function
        """
        x = torch.tensor(x, dtype=torch.float32)  # 将 x 转换为 Tensor
        return torch.exp(-(x ** 2) / (2 * sigma ** 2))

    def forward(self, x, y):
        """
        Calculate the SSIM loss
        :param x: Predict the image, with the shape of (batch_size, channels, height, width)
        :param y: GT image, with the shape of (batch_size, channels, height, width)
        :return: SSIM loss value
        """

        if self.gaussian_kernel.device != x.device:
            self.gaussian_kernel = self.gaussian_kernel.to(x.device)

        # Calculate the mean value
        mu_x = F.conv2d(x, self.gaussian_kernel, padding=self.window_size // 2, groups=self.channels)
        mu_y = F.conv2d(y, self.gaussian_kernel, padding=self.window_size // 2, groups=self.channels)

        # Calculate the variance and covariance
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, self.gaussian_kernel, padding=self.window_size // 2,
                              groups=self.channels) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, self.gaussian_kernel, padding=self.window_size // 2,
                              groups=self.channels) - mu_y_sq
        sigma_xy = F.conv2d(x * y, self.gaussian_kernel, padding=self.window_size // 2, groups=self.channels) - mu_xy

        # SSIM constant
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Calculate SSIM
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
                    (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

        # Return SSIM loss (1 - SSIM)
        return 1 - ssim_map.mean()


class mixLoss(nn.Module):
    def __init__(self):
        super(mixLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.SmoothL1Loss()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.SSIMLoss = SSIMLoss(channels=2)

    def forward(self, y_pred_ND, y_true_ND, y_pred_k, y_true_k):
        #  Calculate MSE
        # mse_loss = self.mse(y_pred_ND, y_true_ND)
        l1_loss = self.l1(y_pred_ND, y_true_ND)
        ssimloss = self.SSIMLoss(y_pred_ND, y_true_ND)
        k_loss = self.CrossEntropyLoss(y_pred_k, y_true_k)
        loss1 = 0.5 * l1_loss + 0.5 * ssimloss
        return loss1 + 3 * k_loss, loss1, k_loss


class NDLoss(nn.Module):
    def __init__(self, channels=2):
        super(NDLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.SmoothL1Loss()

        self.SSIMLoss = SSIMLoss(channels=channels)

    def forward(self, y_pred_ND, y_true_ND):
        # 计算 MSE
        # mse_loss = self.mse(y_pred_ND, y_true_ND)
        l1_loss = self.l1(y_pred_ND, y_true_ND)
        ssimloss = self.SSIMLoss(y_pred_ND, y_true_ND)

        loss1 = 0.5 * l1_loss + 0.5 * ssimloss
        return loss1