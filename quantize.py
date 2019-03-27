import argparse
from utee import misc, quant, selector
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark =True
from collections import OrderedDict
import pprint
import os 

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
parser.add_argument('--type', default='MobileNetV2', help='|'.join(selector.known_models))
parser.add_argument('--data_root', default='~/dataset', help='folder to save the model')
parser.add_argument('--model_root', default='~/pytorch-mobilenet-v2/mobilenetv2_718.pth', help='the path of pre-trained parammeters')
parser.add_argument('--net_root', default='~/pytorch-mobilenet-v2/MobileNetV2.py', help='the path of pre-trained parammeters')

parser.add_argument('--test', type=int, default=1, help='test data distribution')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training')
parser.add_argument('--n_sample',   type=int, default=10, help='number of samples to infer the scaling factor')
parser.add_argument('--gpu', default="0", help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')

parser.add_argument('--replace_bn', type=int, default=0, help='decide if replace bn layer')
parser.add_argument('--map_bn', type=int, default=1, help='decide if map bn layer to conv layer')

parser.add_argument('--input_size', type=int, default=224, help='input size of image')
parser.add_argument('--shuffle', type=int, default=1, help='data shuffle')
parser.add_argument('--overflow_rate', type=float, default=0.0, help='overflow rate')

parser.add_argument('--quant_method', default='linear', help='linear|minmax|log|tanh|scale')
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
#torch.manual_seed(args.seed)
#torch.cuda.manual_seed(args.seed)

# load model and dataset fetcher
if args.use_model_zoo:
    args.model_root = os.path.expanduser('~/.torch/models/')
    model, ds_fetcher, is_imagenet = selector.select(model_name=args.type, model_root=args.model_root)
    args.ngpu = args.ngpu if is_imagenet else 1
else:
    args.model_root = '~/pytorch-mobilenet-v2/mobilenetv2_718.pth'
    model, ds_fetcher  = selector.find(
                        model_name = args.type,
                        model_root = args.model_root,
                        net_root = args.net_root)


# replace bn with 1x1 conv
if args.replace_bn:
    quant.replace_bn(model)


# map bn to conv
if args.map_bn:
    quant.bn2conv(model)

# quantize parameters
print("=================quantize parameters==================")
if args.param_bits < 32:
    state_dict = model.state_dict()
    state_dict_quant = OrderedDict()
    sf_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'running' in k: # quantize bn layer
            #print("k:{}, v:\n{}".format(k,v))
            if args.bn_bits >=32:
                print("Ignoring {}".format(k))
                state_dict_quant[k] = v
                continue
            else:
                bits = args.bn_bits
        else:
            bits = args.param_bits

        if args.quant_method == 'linear':
            sf = bits - 1. - quant.compute_integral_part(v, overflow_rate=args.overflow_rate)
            # sf stands for float bits
            v_quant  = quant.linear_quantize(v, sf, bits=bits)
            #if 'bias' in k:
                #print("{}, sf:{}, quantized value:\n{}".format(k,sf, v_quant.sort(dim=0, descending=True)[0]))
        elif args.quant_method == 'log':
            v_quant = quant.log_minmax_quantize(v, bits=bits)
        elif args.quant_method == 'minmax':
            v_quant = quant.min_max_quantize(v, bits=bits)
        else:
            v_quant = quant.tanh_quantize(v, bits=bits)
        state_dict_quant[k] = v_quant
        print("k={0:<35}, bits={1:<5}, sf={2:d>}".format(k,bits,sf))
    model.load_state_dict(state_dict_quant)
print("======================================================")

# quantize forward activation
print("=================quantize activation==================")
if args.fwd_bits < 32:
    model = quant.duplicate_model_with_quant(model, 
                                                 bits=args.fwd_bits, 
                                                 overflow_rate=args.overflow_rate,
                                                 counter=args.n_sample, 
                                                 type=args.quant_method)

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


# eval model
print("===================eval model=========================")
print(model)
val_ds = ds_fetcher(batch_size=args.batch_size, 
                    data_root=args.data_root, 
                    train=False,
                    val = True,
                    shuffle=args.shuffle,
                    input_size=args.input_size)
if args.test:
    acc1, acc5 = misc.eval_model(model, val_ds, ngpu=args.ngpu, n_sample=1)
else:
    acc1, acc5 = misc.eval_model(model, val_ds, ngpu=args.ngpu)
print("======================================================")


res_str = "type={}, quant_method={}, \n \
          param_bits={}, bn_bits={}, fwd_bits={}, overflow_rate={},\n \
          acc1={:.4f}, acc5={:.4f}".format(
          args.type, args.quant_method, args.param_bits, args.bn_bits, 
          args.fwd_bits, args.overflow_rate, acc1, acc5)
print(res_str)
with open('acc1_acc5.txt', 'a') as f:
    f.write(res_str + '\n')

# show data distribution
if args.test:
    #import visdom 
    import numpy as np
    #viz = visdom.Visdom()
    import matplotlib.pyplot as plt

    '''
    # plot bar
    for k,v in quant.extractor.items():
        #print(k)
        des_v= np.sort(np.abs(v).reshape(-1))
        nums,times = np.unique(des_v,return_counts=True)
        fig = plt.figure()
        ax1 = plt.subplot(111)
        width = 0.2
        print(nums)
        print(times)
        rect = ax1.bar(left=nums,height=times,width=width,color="blue")
        ax1.set_title(k)
        plt.show()
    '''
    '''
    # plot hist
    for k,v in quant.extractor.items():
        #print(k)
        des_v= np.sort(np.abs(v).reshape(-1))
        nums,times = np.unique(des_v,return_counts=True)
        fig = plt.figure()
        print("nums\n{}".format(nums))
        print("times\n{}".format(times))
        plt.hist(des_v, bins=len(nums), density=0, facecolor="blue", edgecolor="black", alpha=0.7)

        plt.xlabel("nums")
        # 显示纵轴标签
        plt.ylabel("times")
        # 显示图标题
        plt.title("nums hist")
        plt.show()
    '''
    
    
    