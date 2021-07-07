import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class Join(nn.Module):

    def __init__(self, params=None, module_name='Default'):
        # TODO:  For now the end module is a case
        # TODO: Make an auto naming function for this.

        super(Join, self).__init__()

        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'mode' not in params:
            raise ValueError(" Missing the mode parameter ")
        if 'after_process' not in params:
            raise ValueError(" Missing the after_process parameter ")
        """" ------------------ IMAGE MODULE ---------------- """
        # Conv2d(input channel, output channel, kernel size, stride), Xavier initialization and 0.1 bias initialization

        self.after_process = params['after_process']
        self.mode = params['mode']

    # TODO: iteration control should go inside the logger, somehow

    def forward(self, x, m):
        # get only the speeds from measurement labels

        if self.mode == 'cat':
            j = torch.cat((x, m), 1)

        else:
            raise ValueError("Mode to join networks not found")

        return self.after_process(j)


class Conv(nn.Module):

    def __init__(self, params=None, module_name='Default'):
        super(Conv, self).__init__()

        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'channels' not in params:
            raise ValueError(" Missing the channel sizes parameter ")
        if 'kernels' not in params:
            raise ValueError(" Missing the kernel sizes parameter ")
        if 'strides' not in params:
            raise ValueError(" Missing the strides parameter ")
        if 'dropouts' not in params:
            raise ValueError(" Missing the dropouts parameter ")
        if 'end_layer' not in params:
            raise ValueError(" Missing the end module parameter ")

        if len(params['dropouts']) != len(params['channels']) - 1:
            raise ValueError("Dropouts should be from the len of channel_sizes minus 1")
        """" ------------------ IMAGE MODULE ---------------- """
        # Conv2d(input channel, output channel, kernel size, stride), Xavier initialization and 0.1 bias initialization

        self.layers = []

        for i in range(0, len(params['channels']) - 1):
            conv = nn.Conv2d(
                in_channels=params['channels'][i],
                out_channels=params['channels'][i + 1],
                kernel_size=params['kernels'][i],
                stride=params['strides'][i]
            )

            dropout = nn.Dropout2d(p=params['dropouts'][i])
            relu = nn.ReLU(inplace=True)
            bn = nn.BatchNorm2d(params['channels'][i + 1])

            layer = nn.Sequential(*[conv, bn, dropout, relu])

            self.layers.append(layer)

        self.layers = nn.Sequential(*self.layers)
        self.module_name = module_name

    def forward(self, x):
        """ Each conv is: conv + batch normalization + dropout + relu """
        x = self.layers(x)

        x = x.view(-1, self.num_flat_features(x))

        return x, self.layers

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_conv_output(self, shape):
        """
           By inputing the shape of the input, simulate what is the ouputsize.
        """

        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat, _ = self.forward(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size


class FC(nn.Module):

    def __init__(self, params=None, module_name='Default'):
        # TODO: Make an auto naming function for this.

        super(FC, self).__init__()
        """" ---------------------- FC ----------------------- """
        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'neurons' not in params:
            raise ValueError(" Missing the kernel sizes parameter ")
        if 'dropouts' not in params:
            raise ValueError(" Missing the dropouts parameter ")
        if 'end_layer' not in params:
            raise ValueError(" Missing the end module parameter ")

        if len(params['dropouts']) != len(params['neurons']) - 1:
            raise ValueError("Dropouts should be from the len of kernels minus 1")

        self.layers = []

        for i in range(0, len(params['neurons']) - 1):

            fc = nn.Linear(params['neurons'][i], params['neurons'][i + 1])
            dropout = nn.Dropout2d(p=params['dropouts'][i])
            relu = nn.ReLU(inplace=True)

            if i == len(params['neurons']) - 2 and params['end_layer']:
                self.layers.append(nn.Sequential(*[fc, dropout]))
            else:
                self.layers.append(nn.Sequential(*[fc, dropout, relu]))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        # if X is a tuple, just return the other elements, the idea is to re pass
        # the intermediate layers for future attention plotting
        if type(x) is tuple:
            return self.layers(x[0]), x[1]
        else:
            return self.layers(x)


class Branching(nn.Module):

    def __init__(self, branched_modules=None):
        """

        Args:
            branch_config: A tuple containing number of branches and the output size.
        """
        # TODO: Make an auto naming function for this.

        super(Branching, self).__init__()
        """ ---------------------- BRANCHING MODULE --------------------- """
        if branched_modules is None:
            raise ValueError("No model provided after branching")

        self.branched_modules = nn.ModuleList(branched_modules)

    # TODO: iteration control should go inside the logger, somehow

    def forward(self, x):
        # get only the speeds from measurement labels

        # TODO: we could easily place this speed outside

        branches_outputs = []
        for branch in self.branched_modules:
            branches_outputs.append(branch(x))

        return branches_outputs
