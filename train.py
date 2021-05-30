# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GicLoss, GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint

from torchpruner.torchpruner.pruner import Pruner 
from torchpruner.torchpruner.attributions import (ShapleyAttributionMetric, WeightNormAttributionMetric)

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images

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

    parser.add_argument("--datamode", default="train")

    parser.add_argument("--stage", default="GMM")
    # parser.add_argument("--stage", default="TOM")

    parser.add_argument("--data_list", default="train_pairs.txt")

    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str,
                        default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=5000)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False, help="Debug mode (no trainig done).")

    opt = parser.parse_args()
    return opt


def train_gmm(opt, train_loader, model, board):
    model.cuda()
    model.train()
        
    # criterion
    criterionL1 = nn.L1Loss()
    gicloss = GicLoss(opt)
    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    finetuning_optimizer = torch.optim.SGD(
        model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
                                                  max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    if not opt.debug:
        _train_gmm(opt, train_loader, model, criterionL1, gicloss, optimizer, board)
    if opt.debug:
        submodel = model.regression.conv
        criteria = (criterionL1, gicloss)
        attribution = WeightNormAttributionMetric(model, train_loader.data_loader, criteria, device=torch.device('cuda'))
        pruner = Pruner(model, input_size=get_GMM_input_size(train_loader), device=torch.device('cuda'), optimizer=finetuning_optimizer)
        layers_of_interest = [layer for layer in submodel.children() if isinstance(layer, torch.nn.modules.conv._ConvNd) or isinstance(layer, nn.BatchNorm2d)]
        num_conv = len([1 for layer in  layers_of_interest if isinstance(layer, torch.nn.modules.conv._ConvNd)])
        for idx, module in enumerate(layers_of_interest):
            if not isinstance(module, nn.Conv2d):
                continue
            num_conv -= 1
            
            if num_conv == 0:
                break #do not prune the last one bc messes up dims

            print("interest layer num:", idx)
            # Compute Weight Value attributions
            attr = attribution.run(module)
            k = int(len(attr) / 10) #10%
            pruning_indices = np.argpartition(attr, k)[:k]

            cascading = layers_of_interest[idx+1:]
            print("cascading layers", cascading)
            pruner.prune_model(module, indices=pruning_indices, cascading_modules=cascading)
            # train for a few epochs
            
            pretty_print_dims(get_pruned_dimensions(submodel))
            _train_gmm(opt, train_loader, model, criterionL1, gicloss, finetuning_optimizer, board, 7000) #14600 / 4 * 2 = 7000

        #carefully finetune prunced model
        pretty_print_dims(get_pruned_dimensions(submodel))
        _train_gmm(opt, train_loader, model, criterionL1, gicloss, finetuning_optimizer, board, 35000) #35000
        torch.save(model, "architectures/pruned_GMM")
    # if opt.debug:
    #     model = torch.load("architectures/pruned_GMM")
    #     pretty_print_dims(get_pruned_dimensions(model.regression.conv))
    #     breakpoint()

def _train_gmm(opt, train_loader, model, criterionL1, gicloss, optimizer, board, num_iter=None):
    if not num_iter: #not finetuning 
        opt.keep_step + opt.decay_step
    for step in range(num_iter):
            print("step:", step)
            iter_start_time = time.time()
            inputs = train_loader.next_batch()

            im = inputs['image'].cuda()
            im_pose = inputs['pose_image'].cuda()
            im_h = inputs['head'].cuda()
            shape = inputs['shape'].cuda()
            agnostic = inputs['agnostic'].cuda()
            c = inputs['cloth'].cuda()
            cm = inputs['cloth_mask'].cuda()
            im_c = inputs['parse_cloth'].cuda()
            im_g = inputs['grid_image'].cuda()
            grid, theta = model(agnostic, cm)    # can be added c too for new training
            warped_cloth = F.grid_sample(c, grid, padding_mode='border')
            warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
            warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

            visuals = [[im_h, shape, im_pose],
                    [c, warped_cloth, im_c],
                    [warped_grid, (warped_cloth+im)*0.5, im]]

            # Lwarp = criterionL1(warped_cloth, im_c)    # loss for warped cloth
            Lwarp = criterionL1(warped_mask, cm)    # loss for warped mask thank xuxiaochun025 for fixing the git code.
            # grid regularization loss
            Lgic = gicloss(grid)
            # 200x200 = 40.000 * 0.001
            Lgic = Lgic / (grid.shape[0] * grid.shape[1] * grid.shape[2])

            loss = Lwarp + 40 * Lgic    # total GMM loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1) % opt.display_count == 0:
                board_add_images(board, 'combine', visuals, step+1)
                board.add_scalar('loss', loss.item(), step+1)
                board.add_scalar('40*Lgic', (40*Lgic).item(), step+1)
                board.add_scalar('Lwarp', Lwarp.item(), step+1)
                t = time.time() - iter_start_time
                print('step: %8d, time: %.3f, loss: %4f, (40*Lgic): %.8f, Lwarp: %.6f' %
                    (step+1, t, loss.item(), (40*Lgic).item(), Lwarp.item()), flush=True)

            if (step+1) % opt.save_count == 0:
                save_checkpoint(model, os.path.join(
                    opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def get_GMM_input_size(train_loader):
    data_sample = train_loader.next_batch()
    agnostic = data_sample['agnostic'].cuda()
    cm = data_sample['cloth_mask'].cuda()
    size = (agnostic.size(), cm.size())
    print(size)
    return size

def get_pruned_dimensions(submodel):
    """
    submodel is the object whose children are layers
    """
    info = dict()
    count = 0
    for module in submodel.children():
        try:
            info[count] = (module.__class__.__name__, module.weight.shape)
        except AttributeError:
            info[count] = (module.__class__.__name__, None)
        count += 1
    return info

def pretty_print_dims(info):
    for key, value in info.items():
        print(f"({key}): {value[0]} {value[1]}")

def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
                                                  max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        pcm = inputs['parse_cloth_mask'].cuda()

        # outputs = model(torch.cat([agnostic, c], 1))  # CP-VTON
        outputs = model(torch.cat([agnostic, c, cm], 1))  # CP-VTON+
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        """visuals = [[im_h, shape, im_pose],
                   [c, cm*2-1, m_composite*2-1],
                   [p_rendered, p_tryon, im]]"""  # CP-VTON

        visuals = [[im_h, shape, im_pose],
                   [c, pcm*2-1, m_composite*2-1],
                   [p_rendered, p_tryon, im]]  # CP-VTON+

        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        # loss_mask = criterionMask(m_composite, cm)  # CP-VTON
        loss_mask = criterionMask(m_composite, pcm)  # CP-VTON+
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            board.add_scalar('L1', loss_l1.item(), step+1)
            board.add_scalar('VGG', loss_vgg.item(), step+1)
            board.add_scalar('MaskL1', loss_mask.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f'
                  % (step+1, t, loss.item(), loss_l1.item(),
                     loss_vgg.item(), loss_mask.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(
                opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))



def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset
    train_dataset = CPDataset(opt)
    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(logdir=os.path.join(opt.tensorboard_dir, opt.name))

    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(
            opt.checkpoint_dir, opt.name, f'gmm_final{"_debug" if opt.debug else ""}.pth'))
    elif opt.stage == 'TOM':
        # model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON
        model = UnetGenerator(
            26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON+
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(
            opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished training %s, named: %s!' % (opt.stage, opt.name))


if __name__ == "__main__":
    main()
