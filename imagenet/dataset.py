import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("/home/jcwang")
import os
import os.path
import numpy as np
import torch
#from torch.utils import data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import torchvision.models as models
from utee import misc
from sequential_imagenet_dataloader import data as lmdb_data

data_root = "~/dataset/"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],       
                                    std=[0.229, 0.224, 0.225])

def get(batch_size=10, data_root=data_root, train=False, val=True, shuffle=True, **kwargs):
    data_root = os.path.expanduser(data_root)
    print("Building IMAGENET data loader, 50000 for test")
    assert train is not True, 'train not supported yet'
    valdir = os.path.join(data_root, 'val')
    #print(data_root)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, 
            transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ])),
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=4, 
        pin_memory=True)
    return val_loader

def ImageNetData(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
    }

    traindir = os.path.join(args.data_root, 'train')
    valdir = os.path.join(args.data_root, 'val')

    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(traindir,
                                                data_transforms['train'])
    image_datasets['val']   = datasets.ImageFolder(valdir,
                                                data_transforms['val'])

    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.workers) for x in ['train', 'val']}


    dataset_sizes = {x: len(image_datasets[x]) / args.batch_size for x in ['train', 'val']}
    return dataloders, dataset_sizes    


def lmdb(args):
    dataloders = {x: lmdb_data.Loader(x,  batch_size=args.batch_size, 
                                shuffle=args.shuffle, 
                                num_workers=args.workers) 
                                for x in ['train', 'val']}
    dataset_sizes = {x: len(dataloders[x]) for x in ['train', 'val']}
    return dataloders, dataset_sizes


def get1(batch_size=10, data_root='~/dataset', train=False, val=True, shuffle=False, **kwargs):
    
    data_root = os.path.expanduser(os.path.join(data_root, 'imagenet-data'))
    print("Building IMAGENET data loader, 50000 for train, 50000 for test")
    ds = []
    assert train is not True, 'train not supported yet'
    if train:
        ds.append(IMAGENET(data_root, batch_size, True, **kwargs))
    if val:
        ds.append(IMAGENET(data_root, batch_size, False, **kwargs))
    ds = ds[0] if len(ds) == 1 else ds
    return ds

class IMAGENET(object):
    def __init__(self, root, batch_size, train=False, input_size=224, **kwargs):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        self.train = train

        if train:
            pkl_file = os.path.join(root, 'train{}.pkl'.format(input_size))
        else:
            pkl_file = os.path.join(root, 'val{}.pkl'.format(input_size))
        self.data_dict = misc.load_pickle(pkl_file)

        self.batch_size = batch_size
        self.idx = 0

    @property
    def n_batch(self):
        return int(np.ceil(self.n_sample* 1.0 / self.batch_size))

    @property
    def n_sample(self):
        return len(self.data_dict['data'])

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.n_batch:
            self.idx = 0
            raise StopIteration
        else:
            img    = self.data_dict['data'  ][self.idx*self.batch_size:(self.idx+1)*self.batch_size].astype('float32')
            target = self.data_dict['target'][self.idx*self.batch_size:(self.idx+1)*self.batch_size]
            self.idx += 1
            return img, target

if __name__ == '__main__':
    get()


