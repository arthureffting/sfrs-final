




from torch import nn


class GeometricModule(nn.Module):

    def __init__(self):
        super(GeometricModule, self).__init__()

    def forward(self, y):
        # y = self.l2(y)
        y = self.l1(y)
        return y