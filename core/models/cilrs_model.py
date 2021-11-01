import importlib

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class CILRSModel(nn.Module):

    def __init__(
        self,
        backbone='resnet18',
        pretrained=True,
        normalize=True,
        num_branch=6,
        speed_dim=1,
        embedding_dim=512,
        hidden_size=256,
        input_speed=True,
        predict_speed=True
    ):
        super().__init__()
        self._normalize = normalize
        assert backbone in ['resnet18', 'resnet34', 'resnet50'], backbone
        backbone_cls = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
        }[backbone]
        self._backbone = backbone_cls(pretrained=pretrained)
        self._backbone.fc = nn.Sequential()
        self._num_branch = num_branch
        self._input_speed = input_speed
        self.predict_speed = predict_speed

        # Project input speed measurement to feature size
        if input_speed:
            self._speed_in = nn.Sequential(
                nn.Linear(speed_dim, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, embedding_dim),
            )

        # Project feature to speed prediction
        if predict_speed:
            self._speed_out = nn.Sequential(
                nn.Linear(embedding_dim, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, speed_dim),
            )

        # Control branches
        fc_branch_list = []
        for i in range(num_branch):
            fc_branch_list.append(
                nn.Sequential(
                    nn.Linear(embedding_dim, hidden_size),
                    nn.ReLU(True),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(True),
                    nn.Linear(hidden_size, 3),
                    nn.Sigmoid(),
                )
            )

        self._branches = nn.ModuleList(fc_branch_list)

    def _normalize_imagenet(self, x):
        """
        Normalize input images according to ImageNet standards.
        :Arguments:
            x (tensor): input images
        """
        x = x.clone()
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        return x

    def encode(self, input_images):
        embedding = 0
        for x in input_images:
            if self._normalize:
                x = self._normalize_imagenet(x)
            embedding += self._backbone(x)
        return embedding

    def forward(self, embedding, speed, command):
        if self._input_speed:
            embedding += self._speed_in(speed.unsqueeze(1))

        control_pred = 0.
        for i, branch in enumerate(self._branches):
            # Choose control for branch of only active command
            # We check for (command - 1) since navigational command 0 is ignored
            control_pred += branch(embedding) * (i == (command.unsqueeze(1) - 1))

        if self.predict_speed:
            speed_pred = self._speed_out(embedding)
            return control_pred, speed_pred

        return control_pred
