import torch
import torch.nn as nn
from . import utils as model_utils


__all__ = ['FCN', 'fcn']


class FCN(nn.Module):

    def __init__(self, frozen=[], num_domains=1, in_channels=3):
        super(FCN, self).__init__()
        self._num_domains = num_domains

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=8, padding="same",
                bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding="same",
                bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding="same",
                bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Global average pooling - gets rid of the time dimension, i.e.
            # giving a fixed-size representation
            # https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721/10
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = 128

    def forward(self, x):
        out = self.layers(x)
        # Remove extra dimension at the end, e.g. to remove the final 1 dimension
        # from (batch_size, global_avg_pooling_out_features, 1)
        out = torch.squeeze(out, 2)
        return out


def fcn(num_domains=1, pretrained=False, **kwargs):
    model = FCN(num_domains=num_domains, **kwargs)
    model_utils.init_weights(model, state_dict=None, num_domains=num_domains, BN2BNDomain=False)
    return model
