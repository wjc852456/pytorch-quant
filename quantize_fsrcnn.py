#import argparse
import sys
from math import log10
from utee import misc, quant, selector
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.backends.cudnn as cudnn
cudnn.benchmark =True
from collections import OrderedDict
import pprint
import os 
import utils
import copy
import tqdm
SR_dataset = os.path.expanduser("~/super-resolution")
sys.path.append(SR_dataset)
from dataset.data import get_test_set, get_test_set_multiple
from torch.utils.data import DataLoader


known_models = [
    'mnist', 'svhn', # 28x28
    'cifar10', 'cifar100', # 32x32
    'stl10', # 96x96
    'alexnet', # 224x224
    'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', # 224x224
    'resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152', # 224x224
    'squeezenet_v0', 'squeezenet_v1', #224x224
    'inception_v3', # 299x299
]

args = utils.prepare_parser()

args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
args.ngpu = len(args.gpu)
misc.ensure_dir(args.logdir) # ensure or create logdir
args.model_root = misc.expand_user(args.model_root)
args.data_root = misc.expand_user(args.data_root)
args.input_size = 299 if 'inception' in args.type else args.input_size
assert args.quant_method in ['linear', 'minmax', 'log', 'tanh', 'scale']
print("=================PARSER==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")


if_CUDA = torch.cuda.is_available()
assert if_CUDA, 'no cuda'
#torch.manual_seed(args.seed)
#torch.cuda.manual_seed(args.seed)

# load model and dataset fetcher
#args.model_root = '~/super-resolution/model_path_all_3x3.pth'
model = torch.load(misc.expand_user(args.model_root))

model_raw = copy.deepcopy(model)
#print(model)
upscale_factor, _ = model.last_part.stride
# replace bn with 1x1 conv
if args.replace_bn:
    quant.replace_bn(model)


# map bn to conv
if args.map_bn:
    quant.bn2conv(model)


# quantize forward activation
print("=================quantize parameters and activations==================")
criterion = torch.nn.MSELoss()
device = torch.device('cuda' if if_CUDA else 'cpu')
def shave(I,border):
    x,y = border
    return I[:,:,x:-x,y:-y]

def test(model, test_loader, upscale_factor, n_sample=None):
    model.eval()
    avg_psnr = 0

    n_sample = len(test_loader) if n_sample is None else n_sample
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(tqdm.tqdm(test_loader, total=n_sample)):
            data, target = data.to(device), target.to(device)
            #print(data)
            prediction = model(data)
            prediction = shave(prediction, [upscale_factor, upscale_factor])
            target =     shave(target, [upscale_factor, upscale_factor])
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
            #progress_bar(batch_num, len(test_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))


    return avg_psnr / len(test_loader)
    #print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(test_loader)))


if args.fwd_bits < 32:
    model = quant.duplicate_model_with_linearquant_nobn(
                                            model, 
                                            param_bits=args.param_bits,
                                            fwd_bits=args.fwd_bits, 
                                            counter=args.n_sample)
    #print(model)
    
    dataset = get_test_set(upscale_factor, args.data_root)
    test_loader = DataLoader(dataset=dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False)
    print("load dataset done")
    test(model, test_loader, upscale_factor, args.n_sample)
print("==============================================================")



print("============================eval==================================")
avg_psnr = test(model, test_loader, upscale_factor)
print("    Average PSNR: {:.4f} dB".format(avg_psnr))

print("==================================================================")
