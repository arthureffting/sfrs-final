import numpy as np
import torch
from torch.autograd import Variable



# input = tensor([x,y,angle,size])
from models.lol.patching.gridder import Gridder


def extract_tensor_patch(img_tensor, input, size=64):
    gridder = Gridder(size)
    grid = gridder.get_grid(img_tensor, input)
    return grid
