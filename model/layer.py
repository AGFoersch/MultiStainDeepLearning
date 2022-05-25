import torch
from torch import nn
import numpy as np


class Fusion(nn.Module):
    def __init__(self, lo_dims, mmhid, dropout_rate, genomic, grad_cam):
        super(Fusion, self).__init__()
        self.genomic = genomic
        self.grad_cam = grad_cam
        if self.grad_cam:
            self.activations = []
            self.gradients = []

        gen_dim = 6 if genomic else 0
        added_dim = int(np.sum(lo_dims))
        self.values = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.ReLU()) for dim in lo_dims])
        self.attention_scores = nn.ModuleList([nn.Linear(added_dim, dim) for dim in lo_dims])
        self.outputs = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(p=dropout_rate)) for dim in lo_dims])

        self.encoder1 = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear((added_dim + gen_dim), mmhid * 2), nn.ReLU(),
                                      nn.Dropout(p=dropout_rate),
                                      nn.Linear(mmhid * 2, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def activations_hook(self, grad):
        self.gradients.append(grad)

    def forward(self, *inputs):
        if self.genomic:
            gen_input = inputs[-1]
            inputs = inputs[:-1]
        cat_vec = torch.cat((inputs), dim=1)
        outputs = []
        for idx,input in enumerate(inputs):
            value = self.values[idx](input)
            attention = nn.Sigmoid()(self.attention_scores[idx](cat_vec))
            output = self.outputs[idx](attention * value)
            if self.grad_cam:
                output.register_hook(self.activations_hook)
                self.activations.append(output)
            outputs.append(output)

        no_kronecker = torch.cat((*outputs, gen_input), dim=1) if self.genomic else torch.cat((outputs), dim=1)
        out = self.encoder1(no_kronecker)
        out = self.encoder2(out)
        return out


class Genomic(nn.Module):
    def __init__(self, dim, gen_dim, mmhid):
        super(Genomic, self).__init__()
        self.linear = nn.Linear((dim+gen_dim), mmhid)

    def forward(self, x1, x2):
        input = torch.cat((x1, x2), dim=1)
        return self.linear(input)
