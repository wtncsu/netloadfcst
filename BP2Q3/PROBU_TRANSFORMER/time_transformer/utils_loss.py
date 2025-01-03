import csv

import torch
import numpy as np


def compute_loss(net: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 loss_function: torch.nn.Module,
                 device: torch.device = 'cpu') -> torch.Tensor:
    """Compute the loss of a network on a given dataset.

    Does not compute gradient.

    Parameters
    ----------
    net:
        Network to evaluate.
    dataloader:
        Iterator on the dataset.
    loss_function:
        Loss function to compute.
    device:
        Torch device, or :py:class:`str`.

    Returns
    -------
    Loss as a tensor with no grad.
    """
    running_loss = 0
    with torch.no_grad():
        for (categorical_var, continuous_var, target_var) in dataloader:

            zone_x = continuous_var[:, :, 0:3].to(device)
            zone_target = target_var[:, :, 0:1]

            net_out = net(zone_x).cpu()
            running_loss += torch.sqrt(loss_function(net_out, zone_target).to(device).cpu())

    return running_loss / len(dataloader)
