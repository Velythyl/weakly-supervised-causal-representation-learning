from nflows.transforms import Transform


class ConditionalAffineScalarTransform(Transform):

    def __init__(self, param_net):
      self.param_net = param_net

    def forward(self, inputs, context=None):
        scale, shift, logabsdet = self.param_net(context)
        outputs = inputs * scale + shift
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        scale, shift, logabsdet = self.param_net(context)
        outputs = (inputs - shift) / scale
        return outputs, -logabsdet
