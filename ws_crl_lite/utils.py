import nflows
from nflows.transforms import Transform
import torch
from torch import nn


class ConditionalAffineScalarTransform(Transform):

    def __init__(self, param_net):
      super().__init__()
      self.param_net = param_net

    def get_scale_and_shift(self, context):

        log_scale_and_shift = self.param_net(context)
        log_scale = log_scale_and_shift[:, 0].unsqueeze(1)
        shift = log_scale_and_shift[:, 1].unsqueeze(1)
        scale = torch.exp(log_scale)

        num_dims = torch.prod(torch.tensor([1]), dtype=torch.float)
        logabsdet = torch.log(scale) * num_dims

        return scale, shift, logabsdet

    def forward(self, inputs, context=None):
        scale, shift, logabsdet = self.get_scale_and_shift(context)
        outputs = inputs * scale + shift
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        scale, shift, logabsdet = self.get_scale_and_shift(context)
        outputs = (inputs - shift) / scale
        return outputs, -logabsdet
