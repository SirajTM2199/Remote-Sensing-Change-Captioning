# Packages

import os
import sys
import json
import random
import numpy as np
from tqdm.auto import tqdm
import torch
from torch import nn
import matplotlib.pyplot as plt


sys.path.append("/home/guest/Documents/Siraj TM/RSCaMa")
from model.model_decoder import DecoderTransformer
from model.model_encoder_attMamba import EnhancedEncoder
from model.model_encoder_attMamba import CrossAttentiveEncoder
from utils_tool.utils import *
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence
from clip import clip
from PIL import Image
from functions import *

# Data
from dataloader import LEVIRMCIDataset_Modified

network='CLIP-ViT-B/32'
clip_model_type = network.replace("CLIP-", "")
clip_model, preprocess = clip.load(clip_model_type,device=device)

Dataset_Path = 'data/LEVIR-MCI-dataset/images'
token_path = 'Change-Agent/Multi_change/data/LEVIR_MCI/tokens/'

test_loader = data.DataLoader(
                LEVIRMCIDataset_Modified(data_folder=Dataset_Path, list_path='Change-Agent/Multi_change/data/LEVIR_MCI/',preprocess=preprocess, split='test', token_folder=token_path, vocab_file='vocab', max_length=42, allow_unk=1),
                batch_size=1, shuffle=True, num_workers=24, pin_memory=True)
