# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, load_checkpoint

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_filters_single_channel_big(layer):
    t = layer.weight.data
    #setting the rows and columns
    nrows = t.shape[0]*t.shape[2]
    ncols = t.shape[1]*t.shape[3]
    
    
    npimg = np.array(t.numpy(), np.float32)
    npimg = npimg.transpose((0, 2, 1, 3))
    npimg = npimg.ravel().reshape(nrows, ncols)
    
    npimg = npimg.T
    
    fig, ax = plt.subplots(figsize=(ncols/10, nrows/200))    
    imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)
    return imgplot.get_figure()

def plot_filters_single_channel(layer):
    t = layer.weight.data
    #kernels depth * number of kernels
    nplots = t.shape[0]*t.shape[1]
    ncols = layer.in_channels
    if ncols < 10:
        ncols = int(np.sqrt(layer.in_channels * layer.out_channels))
    
    nrows = 1 + nplots//ncols
    #convert tensor to numpy image
    npimg = np.array(t.numpy(), np.float32)
    
    count = 0
    fig = plt.figure(figsize=(ncols, nrows))
    
    #looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            print("count:", count)
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
   
    return fig 

#only for three channels?
def plot_filters_multi_channel(layer):
    t = layer.weight.data
    #get the number of kernals
    num_kernels = t.shape[0]    
    
    #define number of columns for subplots
    num_cols = 12
    #rows = num of kernels
    num_rows = num_kernels
    
    #set the figure size
    fig = plt.figure(figsize=(num_cols,num_rows))
    
    #looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        
        #for each kernel, we convert the tensor to numpy 
        npimg = np.array(t[i].numpy(), np.float32)
        #standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        
    plt.savefig('myimage.png', dpi=100)    
    plt.tight_layout()
    plt.show()

def plot_weights(layer, layer_num, opt, counter_container, single_channel = True, collated = False):
  
  #extracting the model features at the particular layer number
  # if opt.name == 'GMM':
  #   layer = model.extractionA.model[layer_num]
  # elif opt.name == 'TOM':
  #   layer = model.model.model[layer_num]
  
  #checking whether the layer is convolution layer or not 
  if isinstance(layer, nn.Conv2d):
    #getting the weight tensor data
    
    if single_channel:
      if collated:
        fig = plot_filters_single_channel_big(layer)
      else:
        fig = plot_filters_single_channel(layer) 
    else:
      if layer.weight.data.shape[1] == 3:
        plot_filters_multi_channel(layer)
      else:
        print("Can only plot weights with three channels with single channel = False")
    
    plt.tight_layout()
    fig.savefig(f'{opt.result_dir}/{opt.name}-layer{counter_container[0]}{"-collated" if opt.collated else ""}.png')
    print("saved figure for conv layer", counter_container[0])
    counter_container[0] += 1
      
  else:
    [plot_weights(c, -1, opt, counter_container, single_channel, collated) for c in layer.children()]

############################################################################################################
#################################### PARSING ###############################################################
############################################################################################################

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="GMM")
    # parser.add_argument("--name", default="TOM")

    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default="data")

    # parser.add_argument("--datamode", default="train")
    parser.add_argument("--datamode", default="test")

    parser.add_argument("--stage", default="GMM")
    # parser.add_argument("--stage", default="TOM")

    # parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--data_list", default="test_pairs.txt")
    # parser.add_argument("--data_list", default="test_pairs_same.txt")

    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)

    parser.add_argument('--tensorboard_dir', type=str,
                        default='tensorboard', help='save tensorboard infos')

    parser.add_argument('--result_dir', type=str,
                        default='weights_visualizations', help='save result infos')

    # parser.add_argument('--checkpoint', type=str, default='checkpoints/GMM/gmm_final.pth', help='model checkpoint for test')
    # parser.add_argument('--checkpoint', type=str, default='checkpoints/TOM/tom_final.pth', help='model checkpoint for test')

    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--collated", type=str2bool, nargs='?', const=True, default=False, help="Debug mode (no trainig done).")
    parser.add_argument("--get_all", type=str2bool, nargs='?', const=True, default=False, help="Debug mode (no trainig done).")
    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')
                        
    opt = parser.parse_args()
    return opt

def main():
    opt = get_opt()
    print(opt)

    checkpoint = f'checkpoints/{opt.name}/{opt.name.lower()}_final.pth'
    layer_num = 0 if not opt.get_all else -1

    if opt.name == 'GMM':
        model = GMM(opt)
        load_checkpoint(model, checkpoint)
        plot_weights(model, layer_num, opt, [0], single_channel = True, collated=opt.collated)
    elif opt.name == 'TOM':
        # model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON
        model = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON+
        load_checkpoint(model, checkpoint)
        #breakpoint()
        plot_weights(model, layer_num, opt, [0], single_channel = True, collated=opt.collated)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

if __name__ == "__main__":
    main()