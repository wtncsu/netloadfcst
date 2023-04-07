import torch
import numpy as np


def compute_val_pinball_score(net: torch.nn.Module,
                          dataloader: torch.utils.data.DataLoader,
                          loss_function: torch.nn.Module,
                          zone_name: str,
                          device: torch.device = 'cuda') -> torch.Tensor:
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
        zone_name:
            Zone description string.
        device:
            Torch device, or :py:class:`str`.

        Returns
        -------
        Pinball score as a tensor with no grad.
    """
    running_loss = 0
    net.eval()

    with torch.no_grad():
        for (categorical_var, continuous_var, target_var) in dataloader:

            if zone_name == 'zone_1':
                zone_x = continuous_var[:, :, 0:3]
                zone_target = target_var[:, :, 0:1]

            elif zone_name == 'zone_2':
                zone_x = continuous_var[:, :, 3:6]
                zone_target = target_var[:, :, 1:2]

            elif zone_name == 'zone_3':
                zone_x = continuous_var[:, :, 6:9]
                zone_target = target_var[:, :, 2:3]

            elif zone_name == 'zone_4':
                zone_x = continuous_var[:, :, 9:12]
                zone_target = target_var[:, :, 3:4]

            elif zone_name == 'zone_5':
                zone_x = continuous_var[:, :, 12:15]
                zone_target = target_var[:, :, 4:5]

            elif zone_name == 'zone_6':
                zone_x = continuous_var[:, :, 15:18]
                zone_target = target_var[:, :, 5:6]

            elif zone_name == 'zone_7':
                zone_x = continuous_var[:, :, 18:21]
                zone_target = target_var[:, :, 6:7]

            elif zone_name == 'zone_8':
                zone_x = continuous_var[:, :, 21:24]
                zone_target = target_var[:, :, 7:8]

            elif zone_name == 'zone_9':
                zone_x = continuous_var[:, :, 24:27]
                zone_target = target_var[:, :, 8:9]

            elif zone_name == 'zone_10':
                zone_x = continuous_var[:, :, 27:30]
                zone_target = target_var[:, :, 9:10]

            elif zone_name == 'zone_11':
                zone_x = continuous_var[:, :, 30:33]
                zone_target = target_var[:, :, 10:11]

            elif zone_name == 'zone_12':
                zone_x = continuous_var[:, :, 33:36]
                zone_target = target_var[:, :, 11:12]

            elif zone_name == 'zone_13':
                zone_x = continuous_var[:, :, 36:39]
                zone_target = target_var[:, :, 12:13]

            elif zone_name == 'zone_14':
                zone_x = continuous_var[:, :, 39:42]
                zone_target = target_var[:, :, 13:14]

            else:
                raise Exception("Incorrect zone name specified, please check !")

            # Propagating through the network
            net_out = net(categorical_var.to(device), zone_x.to(device))[1].cpu()
            running_loss += loss_function(net_out, zone_target).to(device).cpu()

    return running_loss / len(dataloader)