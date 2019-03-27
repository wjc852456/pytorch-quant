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


parser.add_argument('--use_model_zoo', type=int, default=1, help='decide if use model_zoo')
parser.add_argument('--type', default='alexnet', help='|'.join(selector.known_models))
parser.add_argument('--data_root', default='~/dataset/imagenet', help='folder to save the model')
parser.add_argument('--model_root', default='~/.torch/models/', help='the path of pre-trained parammeters')
parser.add_argument('--net_root', default='~/pytorch-mobilenet-v2/MobileNetV2.py', help='the path of pre-trained parammeters')

parser.add_argument('--test', type=int, default=1, help='test data distribution')
parser.add_argument('--batch_size', type=int, default=50, help='input batch size for training')
parser.add_argument('--n_sample',   type=int, default=4, help='number of samples to infer the scaling factor')
parser.add_argument('--gpu', default="0", help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=4, help='number of gpus to use')
parser.add_argument('--logdir', default='./log/default', help='folder to save to the log')

parser.add_argument('--replace_bn', type=int, default=0, help='decide if replace bn layer')
parser.add_argument('--map_bn', type=int, default=1, help='decide if map bn layer to conv layer')

parser.add_argument('--input_size', type=int, default=224, help='input size of image')
parser.add_argument('--shuffle', type=int, default=1, help='data shuffle')
parser.add_argument('--overflow_rate', type=float, default=0.0, help='overflow rate')

parser.add_argument('--quant_method', default='linear', help='linear|minmax|log|tanh|scale')
parser.add_argument('--param_bits', type=int, default=8, help='bit-width for parameters')
parser.add_argument('--fwd_bits',   type=int, default=8, help='bit-width for layer output')
#training param
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--start_epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
parser.add_argument('--print_freq', type=int, default=10)
parser.add_argument('--save_epoch_freq', type=int, default=1)
parser.add_argument('--save_path', type=str, default="./output")
parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")

args = parser.parse_args()
assert torch.cuda.is_available(), 'no cuda'
#os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
misc.ensure_dir(args.logdir)
args.model_root = misc.expand_user(args.model_root)
args.data_root = misc.expand_user(args.data_root)
args.input_size = 299 if 'inception' in args.type else args.input_size
assert args.quant_method in ['linear', 'minmax', 'log', 'tanh', 'scale']
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")



# load model and dataset fetcher
if args.use_model_zoo:
    args.model_root = os.path.expanduser('~/.torch/models/')
    model, ds_fetcher, is_imagenet = selector.select(model_name=args.type, model_root=args.model_root)
    args.ngpu = args.ngpu if is_imagenet else 1
else:
    args.model_root = '~/pytorch-mobilenet-v2/mobilenetv2_718.pth'
    #args.model_root = None
    model, ds_fetcher  = selector.find(
                        model_name = args.type,
                        model_root = args.model_root,
                        net_root = args.net_root)
    if args.model_root is None:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

#model_raw = copy.deepcopy(model)

# replace bn layer
if args.replace_bn:
    quant.replace_bn(model)

# map bn to conv
if args.map_bn:
    quant.bn2conv(model)




device_ids=[int(i) for i in args.gpu]
if args.resume == "":
    model = quant.combine_cb(model, param_bits=args.param_bits, fwd_bits=args.fwd_bits, counter=args.n_sample)
    model = torch.nn.DataParallel(model.cuda(), device_ids=[device_ids[0]])
    #model = model.cuda()
    print(model)


#retrain quantized model
print("===================retrain model======================")
import torch.optim as optim
from torch.optim import lr_scheduler
from imagenet import dataset
import modeltrain

if args.resume != "":
    if os.path.isfile(args.resume):
        print(("=> loading checkpoint '{}'".format(args.resume)))
        checkpoint = torch.load(args.resume, map_location='cpu')
        #checkpoint = torch.load(args.resume)
        #base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.state_dict().items())}
        #model.load_state_dict(base_dict)
        model = checkpoint
        print(model)
    else:
        print(("=> no checkpoint found at '{}'".format(args.resume)))

#model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)

# define loss function
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00004)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.8)

# read data
dataloders, dataset_sizes = dataset.ImageNetData(args)
#dataloders, dataset_sizes = dataset.lmdb(args)
#print("dataset_sizes:{}".format(dataset_sizes))

model = modeltrain.train_model(args=args,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer_ft,
                    scheduler=exp_lr_scheduler,
                    num_epochs=args.num_epochs,
                    dataset_sizes=dataset_sizes,
                    dataloders=dataloders,
                    device_ids=device_ids)
print("======================================================")




'''
print("===================eval model=========================")
#print(model)
if args.test:
    args.batch_size = 1
val_ds = ds_fetcher(batch_size=args.batch_size, 
                    data_root=args.data_root, 
                    train=False,
                    val = True,
                    shuffle=args.shuffle,
                    input_size=args.input_size)
if args.test:
    acc1, acc5 = misc.eval_model(model, val_ds, device_ids=device_ids, n_sample=1)
else:
    acc1, acc5 = misc.eval_model(model, val_ds, device_ids=device_ids)
print("======================================================")

res_str = "type={}, quant_method={}, \n \
          param_bits={}, fwd_bits={},\n \
          acc1={:.4f}, acc5={:.4f}".format(
          args.type, args.quant_method, args.param_bits,
          args.fwd_bits, acc1, acc5)
print(res_str)
'''