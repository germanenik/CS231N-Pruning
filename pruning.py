import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchpruner.pruner import Pruner

def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def get_entropy(M):
    """
    M (n x c) -> 1 x c
    """

def get_indices_to_prune(layer):
    """
    input: conv2d layer
    
    """
    # nn.AvgPool2d: c x h x w -> 1 x c
    # get M of size n x c
    return None
