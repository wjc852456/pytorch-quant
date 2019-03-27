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

parser = argparse.ArgumentParser(description='PyTorch Quantization')

parser.add_argument('--type', default='alexnet', help='|'.join(selector.known_models))
parser.add_argument('--data_root', default='~/dataset', help='folder to save the model')
parser.add_argument('--model_root', default='~/.torch/models/', help='the path of pre-trained parammeters')

parser.add_argument('--test', type=int, default=1, help='test data distribution')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training')
parser.add_argument('--n_sample',   type=int, default=5, help='number of samples to infer the scaling factor')
parser.add_argument('--gpu', default="0", help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')

parser.add_argument('--replace_bn', type=int, default=0, help='decide if replace bn layer')
parser.add_argument('--map_bn', type=int, default=1, help='decide if map bn layer to conv layer')

parser.add_argument('--input_size', type=int, default=224, help='input size of image')
parser.add_argument('--shuffle', type=int, default=1, help='data shuffle')
parser.add_argument('--overflow_rate', type=float, default=0.0, help='overflow rate')

parser.add_argument('--quant_method', default='scale', help='linear|minmax|log|tanh|scale')
parser.add_argument('--param_bits', type=int, default=8, help='bit-width for parameters')
parser.add_argument('--bn_bits',    type=int, default=8, help='bit-width for running mean and std')
parser.add_argument('--fwd_bits',   type=int, default=8, help='bit-width for layer output')
args = parser.parse_args()

args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
args.ngpu = len(args.gpu)
misc.ensure_dir(args.logdir)
args.model_root = misc.expand_user(args.model_root)
args.data_root = misc.expand_user(args.data_root)
args.input_size = 299 if 'inception' in args.type else args.input_size
assert args.quant_method in ['linear', 'minmax', 'log', 'tanh','scale']
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

assert torch.cuda.is_available(), 'no cuda'

# load model and dataset fetcher

args.model_root = os.path.expanduser('~/.torch/models/')
model, ds_fetcher, is_imagenet = selector.select(model_name=args.type, model_root=args.model_root)
args.ngpu = args.ngpu if is_imagenet else 1
model_raw = copy.deepcopy(model)

# replace bn layer
if args.replace_bn:
    quant.replace_bn(model)

# map bn to conv
if args.map_bn:
    quant.bn2conv(model)

'''
print("=================quantize parameters==================")
# quantize parameters
def quantize_param(model):
    if isinstance(model, nn.Sequential):
        for k,v in model._modules.items():
            if isinstance(v,(nn.Conv2d, nn.Linear)):
                v_quant,S,Z = quant.scale_quantize(v.weight, args.param_bits)
                v.weight = Parameter(v_quant)
                v.S = S
                v.Z = Z
            else:
                quantize_param(v)
    else:
        for k,v in model._modules.items():
            quantize_param(v)

if args.param_bits < 32:
    quantize_param(model)
    #for k,v in model.state_dict().items():
    #    print(k)
    #    print(v)
    
print("======================================================")
'''
#print(model)
# quantize forward activation
print("=================quantize activation==================")
if args.fwd_bits < 32:
    model = quant.duplicate_model_with_scalequant(
                model, 
                bits=args.fwd_bits, 
                counter=args.n_sample)

    # ds_fetcher is in path: /imagenet/dataset.get
    val_ds_tmp = ds_fetcher(batch_size=args.batch_size, 
                            data_root=args.data_root, 
                            train=False, 
                            val = True,
                            shuffle=args.shuffle,
                            input_size=args.input_size
                            )
    print("load dataset done")
    misc.eval_model(model, val_ds_tmp, ngpu=1, n_sample=args.n_sample)

print("======================================================")


print("===================eval model=========================")
print(model)
if args.test:
    args.batch_size = 1
else:
    args.batch_size = 50
val_ds = ds_fetcher(batch_size=args.batch_size, 
                    data_root=args.data_root, 
                    train=False,
                    val = True,
                    shuffle=args.shuffle,
                    input_size=args.input_size)
if args.test:
    acc1, acc5 = misc.eval_model(model, val_ds, ngpu=args.ngpu, n_sample=1, model_raw=model_raw)
else:
    acc1, acc5 = misc.eval_model(model, val_ds, ngpu=args.ngpu)
print("======================================================")


res_str = "type={}, quant_method={}, \n \
          param_bits={}, bn_bits={}, fwd_bits={},\n \
          acc1={:.4f}, acc5={:.4f}".format(
          args.type, args.quant_method, args.param_bits, args.bn_bits, 
          args.fwd_bits, acc1, acc5)
print(res_str)

    