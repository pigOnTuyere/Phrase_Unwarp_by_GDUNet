import torch
import time
from tqdm import tqdm
from optparse import OptionParser
import os
import csv

# from network.Unet_1 import UNet
from network.Unet import  UNet
from data.datasetRead import load_data
from utils.trainFunc import train_net, val_net
from utils.myLoss import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.saveLog import save_log





def train(args):
    """train

    """
    dir_model = args.root + "/" + args.output + '/'
    epochs = args.epochs

    time_start = time.time()
    # Use GPU or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Create the model
    net = UNet(input_chanel=3,output_chanel=6).to(device)
    # Load pre-trained weights or not
    if args.weights:
        load_weights = args.weights
        try:
            checkpoint = torch.load(load_weights, map_location='cpu')
            # Load the weights of keyword matching
            net.load_state_dict({k: v for k, v in checkpoint['state_dict'].items() if k in net.state_dict()},
                                strict=False)
            print("Loaded matched weights successfully!")
        except FileNotFoundError:
            print(f"Weights file not found at {load_weights}. Training from scratch.")
        except KeyError:
            print("Key 'state_dict' not found in checkpoint. Training from scratch.")
        except RuntimeError as e:
            print(f"Error loading weights: {e}. Training from scratch.")

    net.train()
    train_loader, val_loader = load_data(batch_size=args.batch_size,data= args["data"])
    # Definition of the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    # Definition of the loss function
    loss_f = NDLoss(channels=6)
    # Define the table header
    header = ['epoch', 'learning rate', 'train loss', "Train MAE" ,  "Train RMSE",'val loss', "val MAE", "Val RMSE",'time cost now/second']
    best_loss = 1000000
    current_lr = args.lr
    # Ready to use the tqdm (A Fast, Extensible Progress Bar for Python and CLI)
    train_losses, train_maes, train_rmses = [], [], []
    val_losses, val_maes, val_rmses = [], [], []

    path_csv = dir_model + "loss and others" + ".csv"

    for epoch in tqdm(range(epochs)):
        # A way of learning rate update
        # if current_lr > 1e-8:
        #     if (epoch + 1) % 10 == 0:
        #         current_lr = current_lr * 0.5
        #     current_lr = max(current_lr, 1e-8)  # Ensure that the learning rate is not lower than 1e-8
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = current_lr  # Update learning rate in optimizer

        # Get training loss function and validating loss function
        train_loss, train_mae,train_rmse= train_net(net, device, train_loader, optimizer, loss_f)
        val_loss, val_mae,val_rmse = val_net(net, device, val_loader, loss_f)

        print(f"Epoch {epoch + 1}/{epochs} - ", f"lr : {optimizer.param_groups[0]['lr']}",
              f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f},Train RMSE: {train_rmse:.4f} - "
              f",Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f},Val RMSE:{val_rmse:.4f}")

        scheduler.step(val_loss)  # #Monitor and verify the losses and adjust the learning rate

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

        # Get time cost now
        time_cost_now = time.time() - time_start
        # Set the values for csv
        values = [epoch + 1, optimizer.param_groups[0]['lr'], train_loss, train_mae,train_rmse, val_loss, val_mae,val_rmse,
                   time_cost_now]
        # Save epoch, learning rate, train loss, val loss and time cost now to a csv

        if os.path.isfile(path_csv) == False:
            file = open(path_csv, 'w', newline='')
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerow(values)
        else:
            file = open(path_csv, 'a', newline='')
            writer = csv.writer(file)
            writer.writerow(values)
        file.close()
        # Save model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'loss': train_loss,
                'optimizer': optimizer.state_dict(),
            }, dir_model + "best_weights" + ".pth")
    time_all = time.time() - time_start
    print("Total time %.4f seconds for training" % (time_all))

def get_args():
    '''

    :return:
    '''

    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=1, type='int', help='batch size')
    parser.add_option('-l', '--learning rate', dest='lr', default=0.05, type='float', help='learning rate')
    parser.add_option('-t', '--weights', dest='pre-trained weight', default=None, help='folder of train data')
    parser.add_option('-i', '--data', dest='data', default='images_14_imgs', help='folder of train data')
    parser.add_option('-r', '--root', dest='root', default="train_my_data", help='output root directory')
    parser.add_option('-s', '--output', dest='output', default='images_sim_16_4', help='folder for model/weights/log')
    (options, args) = parser.parse_args()
    return options

' Run the application '
if __name__ == "__main__":
    args = get_args()
    save_log(args=args, main_func=train, model=UNet)
    train(args)