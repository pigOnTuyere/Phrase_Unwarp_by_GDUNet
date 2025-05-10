import torch

config = {"image_mean": [-0.681],
          "image_std": [0.261],
          "label_mean": [-0.066, 0.2646],
          "label_std": [117.24, 117.28]}

B_min = 5

def denormalize(tensor, device, select_nomalize):
    '''
    denormalize
    :param tensor:
    :param device:
    :param select_nomalize:  Anti-standardization mode selection
    :return:
    '''
    tensor = tensor.float()  # 转换为浮点类型
    if select_nomalize == 1:
        tensor = tensor * torch.tensor(config["label_std"]).view(1, 2, 1, 1).to(device) + \
                 torch.tensor(config["label_mean"]).view(1, 2, 1, 1).to(device)
    elif select_nomalize == 2:
        tensor = tensor * 255.0
    return tensor


def check_Wrap(predict, labels, device,mask=None, B_min=10, select_nomalize=False):
    '''

    :param predict:
    :param labels:
    :param device:
    :param mask:
    :param B_min:
    :param select_nomalize:
    :return:
    '''

    predict = denormalize(predict, device, select_nomalize)
    labels = denormalize(labels, device, select_nomalize)


    assert predict.shape[1] in [2, 6] and labels.shape[1] in [2, 6], "第二个维度不为 2 或 6，检查数据输入格式是否正确"
    assert predict.shape[0] == 1 and labels.shape[0] == 1, "batch size 必须为 1"

    if not torch.is_tensor(mask):
        # Extract sin and cos data
        sinData = labels[:, 0:labels.shape[1]:2, :, :]
        cosData = labels[:, 1:labels.shape[1]:2, :, :]
        
        # Calculate the phase difference
        B = torch.sqrt(sinData ** 2 + cosData ** 2) * 2
        mask = (B > B_min).float()  # 将掩膜转换为 float 类型以便于后续计算


    if predict.shape[1] == 2:
        phase_p = -torch.atan2(predict[:, 0, :, :], predict[:, 1, :, :])
        phase_gt = -torch.atan2(labels[:, 0, :, :], labels[:, 1, :, :]) 
        phase_p = phase_p.unsqueeze(1)
        phase_gt = phase_gt.unsqueeze(1)
    else:  # predict.shape[1] == 6
        phase_p = -torch.atan2(predict[:, 0:6:2, :, :], predict[:, 1:6:2, :, :])
        phase_gt = -torch.atan2(labels[:, 0:6:2, :, :], labels[:, 1:6:2, :, :]) 
    # Calculate sub
    sub = (phase_p - phase_gt)* mask

    return mask, sub



def rmse_wrapped_phase(phase, phase0, mask, is_filter=True):
    '''
    Calculate the root mean square error (RMSE) between the two phases and limit the difference within the range of [-π, π].
    :param phase: predicted wrap phrase
    :param phase0: GT wrap phrase
    :param mask:
    :param is_filter: Whether to filter out abnormal pixel values
    :return:
    - rmse: wrap phrase rmse
    - count: The number of abnormal pixel values filtered
    '''


    mask = mask.to(phase.device).float()


    phase *= mask
    phase0 *= mask

    # Calculate the number of available pixels
    num_pixels = torch.sum(mask)

    # Calculate the phase difference
    e = phase - phase0
    count = torch.sum(torch.abs(e) > 5).item()

    # Set the phase difference with an absolute value greater than 5 to 0
    if is_filter:
        e = torch.where(torch.abs(e) > 5, torch.tensor(0.0, device=e.device), e)

    # Calculate rmse
    rmse = torch.sqrt(torch.sum(e ** 2) / (num_pixels - count * is_filter + 1e-6))
    return rmse, count

def wrap_RMSE(predict, labels, device,mask=None, B_min=10, select_nomalize=0):
    # denormalize
    predict = denormalize(predict, device, select_nomalize)
    labels = denormalize(labels, device, select_nomalize)

    # b c H W
    assert (predict.shape[1] == 2 or predict.shape[1] == 6) and (labels.shape[1] == 2 or labels.shape[1] == 6), \
        "第二个维度不正确，检查数据输入格式是否正确"

    # If the mask of the object area is not provided, calculate the mask of the object area by self
    if not torch.is_tensor(mask):
        #  Extract sin and cos data
        sinData = labels[:, 0:labels.shape[1]:2, :, :]
        cosData = labels[:, 1:labels.shape[1]:2, :, :]
        
        # Calculate B and the mask
        B = torch.sqrt(sinData ** 2 + cosData ** 2) * 2
        mask = (B > B_min).float()  #

    # Calculate wrap Phrase
    if predict.shape[1] == 2:
        phase_p = -torch.atan2(predict[:, 0, :, :], predict[:, 1, :, :])
        phase_gt = -torch.atan2(labels[:, 0, :, :], labels[:, 1, :, :])
        phase_p = phase_p.unsqueeze(1)
        phase_gt = phase_gt.unsqueeze(1)  # 保持tensor格式一致
    elif predict.shape[1] == 6:
        phase_p = -torch.atan2(predict[:, 0:6:2, :, :], predict[:, 1:6:2, :, :])  # 选择索引为 (0, 2, 4) 的通道
        phase_gt = -torch.atan2(labels[:, 0:6:2, :, :], labels[:, 1:6:2, :, :])  # 选择索引为 (0, 2, 4) 的通道

    # Calculate wrap Phrase RMSE
    rmse_values, count = rmse_wrapped_phase(phase_p, phase_gt, mask)

    return torch.mean(rmse_values), count

def RMSE(predict, labels, device,mask=None, B_min=10, select_nomalize=0):
   
    # b c H W
    predict = denormalize(predict, device, select_nomalize)
    labels = denormalize(labels, device, select_nomalize)
    # print(f"predict shape: {predict.shape}")
    # print(f"labels shape: {labels.shape}")
    # print(f"mask shape: {mask.shape}")
    e = (labels- predict)*mask
    num_pixels = torch.sum(mask)
    rmse = torch.sqrt(torch.sum(e ** 2) / (num_pixels + 1e-6))  # 防止除以零
    return rmse


def MAE(predict, labels, device,mask, select_nomalize=False):
    predict = denormalize(predict, device, select_nomalize)
    labels = denormalize(labels, device, select_nomalize)
    # print(f"predict shape: {predict.shape}")
    # print(f"labels shape: {labels.shape}")
    # print(f"mask shape: {mask.shape}")
    mae = torch.sum(torch.abs(labels - predict)*mask)/(torch.sum(mask)*mask.size()[1])
    return mae