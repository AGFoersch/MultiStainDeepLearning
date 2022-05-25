import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, num_classes=None):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def freeze(self, until:-1):
        for c in list(self.children())[:until]:
            for param in c.parameters():
                param.requires_grad = False

        params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.parameters())])
        print(f'Trainable parameters: {params}')