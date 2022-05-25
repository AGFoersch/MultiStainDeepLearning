import torch.nn as nn
from base import BaseModel
import torchvision
import model.layer as module_layer
import torch
from collections import OrderedDict


class MultiModel(BaseModel):
    def __init__(self, num_classes, lo_dims, lo_pretrained=None, mmhid=64, dropout_rate=0.25, genomic=False,
                 grad_cam:bool=False):
        super(MultiModel, self).__init__(num_classes=num_classes)
        self.lo_dims = lo_dims
        self.models = nn.ModuleList([])

        self.grad_cam = grad_cam

        if self.grad_cam:
            self.activations = []
            self.gradients = []

        for idx,dim in enumerate(lo_dims):
            model = torchvision.models.resnet18(pretrained=True)
            model.fc = nn.Linear(512, dim)

            named_children = list(model.named_children())
            first_part = nn.Sequential(OrderedDict(named_children[:-2]))
            second_part = nn.Sequential(
                OrderedDict([named_children[-2], ('flatten', nn.Flatten()), named_children[-1]]))

            self.models.append(nn.Sequential(OrderedDict([('activations',first_part), ('features', second_part)])))

        if genomic:
            self.models.append(nn.Identity())

        if lo_pretrained is not None:
            for idx,pretrained in enumerate(lo_pretrained):
                model_dict = self.models[idx].state_dict()
                checkpoint = torch.load(pretrained)
                short_pretrain = {'.'.join(k.split('.')[2:]): v for k, v in checkpoint['state_dict'].items() if 'model' in k}
                pretrained_dict = {k: v for k, v in short_pretrain.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[idx].load_state_dict(model_dict)
                print(f'pretrained weights loaded for model no. {idx}')

        if len(lo_dims) > 1:
            self.fusion = module_layer.Fusion(lo_dims, mmhid, dropout_rate, genomic, grad_cam)
        else:
            self.fusion = nn.Linear(*lo_dims, mmhid) if not genomic else module_layer.Genomic(*lo_dims, 6, mmhid)
        self.classifier = nn.Linear(mmhid, num_classes)

    def activations_hook(self, grad):
        self.gradients.append(grad)

    def get_last_activ_layers(self):
        return [self.models[i].activations.layer4[1] for i in range(len(self.lo_dims))]

    def get_fusion_output_layers(self):
        if len(self.lo_dims) > 1:
            return [layer for layer in self.fusion.outputs]
        else:
            return None

    # regular_forward contains the stuff that was formerly in this method.
    # I'm organizing things this way so I can switch out the forward method on the fly when needed.
    def forward(self, *inputs):
        return self.regular_forward(*inputs)

    def regular_forward(self, *inputs):
        if self.grad_cam:
            self.activations = activations = [model.activations(input) for model,input in zip(self.models, inputs)]
            if self.grad_cam:
                for act in activations:
                    act.register_hook(self.activations_hook)
            outputs = [model.features(act) for model,act in zip(self.models, activations)]
        else:
            outputs = [model(input) for model,input in zip(self.models, inputs)]

        kronecker = self.fusion(*outputs)
        hazard = self.classifier(kronecker)
        return hazard

    def forward_for_impact_survival(self, *inputs):
        """
        More or less like regular_forward, but sigmoids the result and additionally
        returns 1-result.
        """
        sigmoided_output = torch.sigmoid(self.regular_forward(*inputs))
        out_tensor = torch.zeros((sigmoided_output.shape[0], 2), requires_grad=True).to(sigmoided_output.device)
        out_tensor[:,0] = sigmoided_output[:,0]
        out_tensor[:,1] = 1 - sigmoided_output[:,0]
        return out_tensor

    def forward_for_gradcam_survival(self, backprop_tensor, backprop_mod_idx, other_tensors):
        """
        Forward method for captum's guided gradcam class.

        backprop_tensor:    The tensor to calculate the attributions for. This tensor should have
                            grad_enabled == True.
        backprop_mod_idx:   Modality index of the attribution tensor. Basically, this should be the
                            position (0-indexed) at which backprop_tensor usually gets passed to
                            forward.
        other_tensors:      List of (mod_idx, tensor) tuples for the other model inputs. mod_idx
                            is again the 0-indexed position at which tensor usually gets passed to
                            forward.

        None of the indices here are sanity-checked, so be sure to pass them right!
        """
        inputs = [None for i in range(len(other_tensors)+1)]
        inputs[backprop_mod_idx] = backprop_tensor
        for idx, tensor in other_tensors:
            inputs[idx] = tensor
        sigmoided_output = torch.sigmoid(self.regular_forward(*inputs)) # shape (bs, 1)
        device = sigmoided_output.device
        out_tensor = torch.zeros((sigmoided_output.shape[0], 2), requires_grad=True).to(device)
        out_tensor[:,0] = sigmoided_output[:,0]
        out_tensor[:,1] = 1 - sigmoided_output[:,0]
        return out_tensor

    def forward_for_gradcam_classification(self, backprop_tensor, backprop_mod_idx, other_tensors):
        """
        Same arguments as the survival-method
        """
        inputs = [None for i in range(len(other_tensors)+1)]
        inputs[backprop_mod_idx] = backprop_tensor
        for idx, tensor in other_tensors:
            inputs[idx] = tensor
        return self.regular_forward(*inputs)

