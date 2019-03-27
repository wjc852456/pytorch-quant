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
from imagenet import dataset
import torch.optim as optim
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
parser.add_argument('--data_root', default="/home/jcwang/dataset/imagenet", help='folder to save the model')
parser.add_argument('--model_root', default='~/.torch/models/', help='the path of pre-trained parammeters')
parser.add_argument('--net_root', default='~/pytorch-mobilenet-v2/MobileNetV2.py', help='the path of pre-trained parammeters')

parser.add_argument('--test', type=int, default=0, help='test data distribution')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
parser.add_argument('--n_sample',   type=int, default=10, help='number of samples to infer the scaling factor')
parser.add_argument('--gpu', default="1", help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=4, help='number of gpus to use')
parser.add_argument('--logdir', default='./default', help='folder to save to the log')

parser.add_argument('--replace_bn', type=int, default=0, help='decide if replace bn layer')
parser.add_argument('--map_bn', type=int, default=1, help='decide if map bn layer to conv layer')

parser.add_argument('--input_size', type=int, default=224, help='input size of image')
parser.add_argument('--shuffle', type=int, default=1, help='data shuffle')
parser.add_argument('--overflow_rate', type=float, default=0.0, help='overflow rate')

parser.add_argument('--quant_method', default='linear', help='linear|minmax|log|tanh|scale')
parser.add_argument('--param_bits', type=int, default=2, help='bit-width for parameters')
parser.add_argument('--fwd_bits',   type=int, default=8, help='bit-width for layer output')

parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--lr_pre', type=float, default=0.01)
parser.add_argument('--lr_crr', type=float, default=0.001)
parser.add_argument('--rho', type=float, default=1.0)
parser.add_argument('--num_epochs', type=int, default=1)
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
if args.use_model_zoo:
    args.model_root = os.path.expanduser(args.model_root)
    model, ds_fetcher, is_imagenet = selector.select(model_name=args.type, model_root=args.model_root)
    args.ngpu = args.ngpu if is_imagenet else 1
else:
    args.model_root = '~/pytorch-mobilenet-v2/mobilenetv2_718.pth'
    args.type = "MobileNetV2"
    model, ds_fetcher  = selector.find(
                        model_name = args.type,
                        model_root = args.model_root,
                        net_root = args.net_root)


q_model = copy.deepcopy(model)

# replace bn layer
#if args.replace_bn:
#    quant.replace_bn(model)

# map bn to conv
if args.map_bn:
    quant.bn2conv(model)

# add alpha to each sub module
def add_para(m):
    if isinstance(m,(nn.Conv2d,nn.Linear)):
        lambd = torch.randn_like(m.weight)
        m.register_buffer("lambd",lambd)
        alpha = torch.abs(torch.randn(1))
        m.register_buffer("alpha",alpha)
        re_loss = torch.tensor(0)
        m.register_buffer("re_loss",re_loss)

        l_shift = [2**i for i in range(args.param_bits+1)]
        Q_set = torch.FloatTensor(l_shift)
        Q_set = torch.cat((Q_set,-Q_set))
        Q_set = torch.cat((Q_set,torch.FloatTensor([0])))
        Q_set = Q_set.unsqueeze(1)
        m.register_buffer("Q_set",Q_set)

        w = m.weight.view(-1)
        Q_set_expand = Q_set.expand(len(Q_set),len(w))
        G_set_expand = alpha * Q_set_expand
        #print(w.type())
        _, min_id = torch.min( torch.abs(G_set_expand-w),0 )

        Q = Q_set_expand[min_id,torch.arange(len(min_id))]
        m.register_buffer("Q",Q.reshape_as(m.weight))



def update_alpha(m):
    if isinstance(m, (nn.Conv2d,nn.Linear)):
        #V = m.weight.view(-1)+m.lambd.view(-1)
        #Q = m.Q.view(-1)
        #m.alpha = ( V.unsqueeze(0).mm(Q.unsqueeze(1)) ) / ( Q.unsqueeze(0).mm(Q.unsqueeze(1)) )
        V = m.weight+m.lambd
        Q = m.Q
        m.alpha = torch.sum(V*Q) / torch.sum(Q*Q)

def update_Q(m):
    if isinstance(m, (nn.Conv2d,nn.Linear)):
        w = m.weight.view(-1)
        V = m.weight.view(-1) + m.lambd.view(-1)
        Q_ = V / m.alpha
        Q_set_expand = m.Q_set.expand(len(m.Q_set),len(w))
        _, min_id = torch.min( torch.abs(Q_set_expand-Q_),0 )
        Q = Q_set_expand[min_id,torch.arange(len(min_id))]
        m.Q = Q.reshape_as(m.weight)

def update_lambd(m):
    if isinstance(m, (nn.Conv2d,nn.Linear)):
        m.lambd = m.lambd + m.weight - m.alpha*m.Q

def admm_loss(m):
    if isinstance(m,(nn.Conv2d,nn.Linear)):
        G = m.alpha * m.Q
        re_loss = admm_criterion(m.weight+m.lambd, G)
        m.re_loss = re_loss
        #print("re_loss:{}".format(re_loss))
 
def param_map(model,q_model):
    for m1,m2 in zip(model.modules(),q_model.modules()):
        if isinstance(m1,(nn.Conv2d,nn.Linear)):
            m2.weight = Parameter(m1.alpha*m1.Q)

model.apply(add_para)

#model.cuda()
device_ids=[int(i) for i in args.gpu]


if args.test:
    args.batch_size = 1
dataloders, dataset_sizes = dataset.ImageNetData(args)

admm_criterion = nn.MSELoss()



print("===================ADMM quant=========================")
criterion = nn.CrossEntropyLoss()
optimizer_pre = optim.SGD(model.parameters(), lr=args.lr_pre, momentum=0.9)
optimizer_crr = optim.SGD(model.parameters(), lr=args.lr_crr, momentum=0.9)

for epoch in range(0, args.num_epochs):
    running_corrects = 0
    for i, (inputs,labels) in enumerate(dataloders['train']):
        #inputs = inputs.cuda()
        #labels = labels.cuda()

        state = copy.deepcopy(model.state_dict())
        # prediction step 
        optimizer_pre.zero_grad()
        re_loss = 0
        outputs = model(inputs)
        #_, preds = torch.max(outputs.data, 1)
        model.apply(admm_loss)
        for name, para in model.state_dict().items():
            if "re_loss" in name:
                re_loss += para
        loss = criterion(outputs, labels) + 0.5*args.rho*re_loss
        #print("loss prediction:{}".format(loss))
        loss.backward()
        optimizer_pre.step()
        # correction step
        optimizer_crr.zero_grad()
        re_loss = 0
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        model.apply(admm_loss)
        for name, para in model.state_dict().items():
            if "re_loss" in name:
                re_loss += para
                #print("re_loss:{}".format(re_loss))
        loss = criterion(outputs, labels) + 0.5*args.rho*re_loss
        #running_corrects += torch.sum(preds == labels.data)
        #batch_acc = float(running_corrects) / ((i+1)*args.batch_size)
        #print('Acc :{:.4f}'.format(batch_acc))
        #print("loss correction:{}".format(loss))
        loss.backward()
        model.load_state_dict(state)
        optimizer_crr.step()

        
        # update alpha and q-weight
        for j in range(0,10):
            #pass
            # optimize alpha with Q fixed
            model.apply(update_alpha)
            # optimize Q with alpha fixed
            model.apply(update_Q)
        model.apply(update_lambd)

        param_map(model,q_model)
        outputs = q_model(inputs)
        _, preds = torch.max(outputs.data, 1)
        running_corrects += torch.sum(preds == labels.data)
        batch_acc = float(running_corrects) / ((i+1)*args.batch_size)
        print('Acc :{:.4f}'.format(batch_acc))
        #if(i==0):
        #    break

print("======================================================")




print("===================eval model=========================")
#for name,module in model.state_dict().items():
#    print(name)
#    print(module)
#print(model)

'''
val_ds = dataloders['val']

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