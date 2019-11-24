import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN,self).__init__()

        # convolutional part: 3 layers + ReLU
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # use conv_out_size to get the exact shape of output from conv part
        conv_out_size = self._get_conv_out(input_shape)

        # linear part: 2 fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        '''
        Uses the input shape to get the exact output shape of the conv. part
        Applies the conv layers to a fake sensor of the same size as the actual input
        This only happens once
        '''
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        '''
        Accepts 4D input tensor (batch_size, color_channel, x_dim, y_dim)
        The color channel is the stack of sub-sequent frames
        First: conv on the input -> 4D tensor output
        Output flattened to 2D (batch_size, parameters returned by the conv for this batch
        as one long vector of numbers)
        Finally: pass the flattened 2D tensor to FC layers -> Q-values for every batch input
        '''
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

    