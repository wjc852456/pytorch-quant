import argparse
from utee import misc, quant, selector
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark =True
from collections import OrderedDict
import pprint
import os 
import torch.nn as nn
from torch.nn.parameter import Parameter
import copy



model_root = '~/pytorch-mobilenet-v2/mobilenetv2_718.pth'
net_root = '~/pytorch-mobilenet-v2/MobileNetV2.py'
model, _  = selector.find(model_name = "MobileNetV2",model_root = model_root,net_root = net_root)

resume = "/home/jcwang/pytorch-quant/output_8bits/epoch_10.pth.tar"


if os.path.isfile(resume):
    print(("=> loading checkpoint '{}'".format(resume)))
    checkpoint = torch.load(resume, map_location='cpu')
    model = checkpoint
    print(model)
else:
    print("fuck off")
