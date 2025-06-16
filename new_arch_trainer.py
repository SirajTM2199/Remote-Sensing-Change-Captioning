# Imports
import torch
import numpy as np
import os
import json
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
from clip import clip
import sys
sys.path.append('/home/guest/Documents/Siraj TM/RSCaMa')
from model.model_decoder import DecoderTransformer
import torchvision.transforms.functional as TF
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.layers import drop_path, trunc_normal_
from PIL import Image
from data.LEVIR_CC.LEVIRCC_Modified import LEVIRCCDataset

#
