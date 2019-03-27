import argparse
from utee import misc, quant, selector
import torch

parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
parser.add_argument('--type', default='alexnet', help='|'.join(selector.known_models))
parser.add_argument('--model_root', default='/home/jcwang/.torch/models/', help='folder to load the model')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--data_root', default='~/dataset', help='folder to save the model')
parser.add_argument('--shuffle', type=int, default=0, help='data shuffle')
parser.add_argument('--input_size', type=int, default=224, help='input size of image')
#parser.add_argument('--seed', type=int, default=117, help='random seed')
args = parser.parse_args()
# don not set seed if you want random input !!!
#torch.manual_seed(args.seed)
#torch.cuda.manual_seed(args.seed)

print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

model_raw, ds_fetcher, is_imagenet = selector.select(model_name=args.type, model_root=args.model_root)
val_dataloder = ds_fetcher(batch_size=args.batch_size, 
                        data_root=args.data_root, 
                        train=False, 
                        val = True,
                        shuffle=args.shuffle,
                        input_size=args.input_size
                        )
model_raw = model_raw.cuda()
model_raw = model_raw.eval()
#val_data,val_target = iter(val_dataloder).next()
val_data = torch.randn(args.batch_size,3,224,224)
val_data = val_data.cuda()
model_raw(val_data)
state_dict = model_raw.state_dict()
for k,v in state_dict.items():
    if "running" in k:
        print(k)
        print(v)