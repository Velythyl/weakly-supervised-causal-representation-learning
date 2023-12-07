import nflows
from nflows.transforms import Transform
from torch import nn


class ConditionalAffineScalarTransform(Transform):

    def __init__(self, param_net):
      super().__init__()
      self.param_net = param_net

    def forward(self, inputs, context=None):
        scale, shift, logabsdet = self.param_net(context)
        outputs = inputs * scale + shift
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        scale, shift, logabsdet = self.param_net(context)
        outputs = (inputs - shift) / scale
        return outputs, -logabsdet

# class ParamsNet(nn.Module):

def affine_transform(dim):
    mlps = []
    for di in range(dim):
        mlps.append(
            nn.Sequential(
                nn.Linear
            )
        )
