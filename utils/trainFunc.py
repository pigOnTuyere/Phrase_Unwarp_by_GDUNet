import torch
from utils.evaluationIndex import *
import torch.nn.functional as F  # 用于 MAE 计算


def train_net(model, device, loader, optimizer, loss_f):
    """

    :param model:
    :param device:
    :param loader:
    :param optimizer:
    :return:
    """

    model.train()

    train_loss, train_mae = 0, 0
    train_Wrap_RMSE = 0
    for inputs, targets, _ in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        mask = inputs[:, 0, :, :] > 0
        targets = targets.squeeze(1)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_f(outputs, targets)
        mae = F.l1_loss(outputs, targets)

        rmse, _ = wrap_RMSE(outputs, targets, mask=mask.unsqueeze(1).float(), select_nomalize=0, device=device)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_mae += mae.item()
        train_Wrap_RMSE += rmse.item()

    train_loss /= len(loader)
    train_mae /= len(loader)

    train_Wrap_RMSE /= len(loader)

    return train_loss, train_mae, train_Wrap_RMSE


def val_net(model, device, loader, loss_f):
    """

    :param model:
    :param device:
    :param loader:
    :return:
    """

    model.eval()
    val_loss, val_mae = 0, 0
    val_Wrap_RMSE = 0
    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            mask = inputs[:, 0, :, :] > 0
            targets = targets.squeeze(1)

            outputs = model(inputs)

            loss = loss_f(outputs, targets)
            mae = F.l1_loss(outputs, targets)
            rmse, _ = wrap_RMSE(outputs, targets, mask=mask.unsqueeze(1).float(), select_nomalize=0, device=device)

            val_loss += loss.item()
            val_mae += mae.item()
            val_Wrap_RMSE += rmse.item()

    val_loss /= len(loader)
    val_mae /= len(loader)
    val_Wrap_RMSE /= len(loader)

    return val_loss, val_mae, val_Wrap_RMSE