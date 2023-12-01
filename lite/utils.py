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

class AffineTransformZ2x(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        def get_net():
            return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
        )

        self.nets = [get_net() for _ in range(dim)]

    def forward(self, input):
        for i, inp in enumerate(input):




        shift_scale = self.net(input)
        shift = shift_scale[:,0]
        scale = shift_scale[:,1]
        return input * scale + shift

