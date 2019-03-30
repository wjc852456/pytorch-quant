from utee import misc
import os
from imagenet import dataset
#print = misc.logger.info
from IPython import embed
import sys
import torch


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

def mnist(cuda=True, model_root=None):
    print("Building and initializing mnist parameters")
    from mnist import model, dataset
    m = model.mnist(pretrained=os.path.join(model_root, 'mnist.pth'))
    if cuda:
        m = m.cuda()
    return m, dataset.get, False

def svhn(cuda=True, model_root=None):
    print("Building and initializing svhn parameters")
    from svhn import model, dataset
    m = model.svhn(32, pretrained=os.path.join(model_root, 'svhn.pth'))
    if cuda:
        m = m.cuda()
    return m, dataset.get, False

def cifar10(cuda=True, model_root=None):
    print("Building and initializing cifar10 parameters")
    from cifar import model, dataset
    m = model.cifar10(128, pretrained=os.path.join(model_root, 'cifar10.pth'))
    if cuda:
        m = m.cuda()
    return m, dataset.get10, False

def cifar100(cuda=True, model_root=None):
    print("Building and initializing cifar100 parameters")
    from cifar import model, dataset
    m = model.cifar100(128, pretrained=os.path.join(model_root, 'cifar100.pth'))
    if cuda:
        m = m.cuda()
    return m, dataset.get100, False

def stl10(cuda=True, model_root=None):
    print("Building and initializing stl10 parameters")
    from stl10 import model, dataset
    m = model.stl10(32, pretrained=os.path.join(model_root, 'stl10.pth'))
    if cuda:
        m = m.cuda()
    return m, dataset.get, False

def alexnet(cuda=False, model_root=None):
    print("Building and initializing alexnet parameters")
    from imagenet import alexnet as alx
    m = alx.alexnet(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def vgg16(cuda=False, model_root=None):
    print("Building and initializing vgg16 parameters")
    from imagenet import vgg
    m = vgg.vgg16(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def vgg16_bn(cuda=False, model_root=None):
    print("Building vgg16_bn parameters")
    from imagenet import vgg
    m = vgg.vgg16_bn(model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def vgg19(cuda=False, model_root=None):
    print("Building and initializing vgg19 parameters")
    from imagenet import vgg
    m = vgg.vgg19(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def vgg19_bn(cuda=False, model_root=None):
    print("Building vgg19_bn parameters")
    from imagenet import vgg
    m = vgg.vgg19_bn(model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def inception_v3(cuda=False, model_root=None):
    print("Building and initializing inception_v3 parameters")
    from imagenet import inception
    m = inception.inception_v3(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def resnet18(cuda=False, model_root=None):
    print("Building and initializing resnet-18 parameters")
    from imagenet import resnet
    m = resnet.resnet18(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def resnet34(cuda=False, model_root=None):
    print("Building and initializing resnet-34 parameters")
    from imagenet import resnet
    m = resnet.resnet34(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def resnet50(cuda=False, model_root=None):
    print("Building and initializing resnet-50 parameters")
    from imagenet import resnet
    m = resnet.resnet50(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def resnet101(cuda=False, model_root=None):
    print("Building and initializing resnet-101 parameters")
    from imagenet import resnet
    m = resnet.resnet101(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def resnet152(cuda=False, model_root=None):
    print("Building and initializing resnet-152 parameters")
    from imagenet import resnet
    m = resnet.resnet152(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def squeezenet_v0(cuda=False, model_root=None):
    print("Building and initializing squeezenet_v0 parameters")
    from imagenet import squeezenet
    m = squeezenet.squeezenet1_0(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def squeezenet_v1(cuda=False, model_root=None):
    print("Building and initializing squeezenet_v1 parameters")
    from imagenet import squeezenet
    m = squeezenet.squeezenet1_1(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def select(model_name, **kwargs): # return one of the functions above
    assert model_name in known_models, model_name
    kwargs.setdefault('model_root', os.path.expanduser('~/.torch/models'))
    return eval('{}'.format(model_name))(**kwargs)

def find(model_name, model_root, net_root, **kwargs):
    if net_root==None:
        model = torch.load(os.path.expanduser(model_root))
    else:
        root_list = net_root.split('/')
        path = '/'.join(root_list[0:-1])
        sys.path.append(os.path.expanduser(path))
        exec('from {} import {}'.format(root_list[-1][:-3], model_name))
        model = eval('{}()'.format(model_name))
    if model_root==None:
        pass
    elif net_root!=None:
        model.load_state_dict(torch.load(os.path.expanduser(model_root)))
    return model, dataset.get

if __name__ == '__main__':
    m1 = alexnet()
    embed()


