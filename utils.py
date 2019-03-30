import argparse
from utee import misc, quant, selector
def prepare_parser():
  parser = argparse.ArgumentParser(description='PyTorch Quantization')

  parser.add_argument('--use_model_zoo', default=False, help='decide if use model_zoo')
  parser.add_argument('--type', default='FSRCNN', help='|'.join(selector.known_models))
  parser.add_argument('--data_root', default='~/super-resolution/dataset/test/Set5', help='folder to for dataset')
  parser.add_argument('--model_root', default='~/super-resolution/model_path_all_3x3.pth', help='the path of pre-trained parammeters')
  parser.add_argument('--net_root', default=None, help='the path of net config')

  parser.add_argument('--test', default=False, help='test data distribution')
  parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training')
  parser.add_argument('--n_sample',   type=int, default=2, help='number of samples to infer the scaling factor')
  parser.add_argument('--gpu', default="0", help='index of gpus to use')
  parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
  parser.add_argument('--logdir', default='log/default', help='folder to save to the log')

  parser.add_argument('--replace_bn', default=False, help='decide if replace bn layer')
  parser.add_argument('--map_bn', default=False, help='decide if map bn layer to conv layer') 

  parser.add_argument('--input_size', type=int, default=224, help='input size of image')
  parser.add_argument('--shuffle', default=True, help='data shuffle')
  parser.add_argument('--overflow_rate', type=float, default=0.0, help='overflow rate')

  parser.add_argument('--quant_method', default='linear', help='linear|minmax|log|tanh|scale')
  parser.add_argument('--param_bits', type=int, default=8, help='bit-width for parameters')
  #parser.add_argument('--bn_bits',    type=int, default=8, help='bit-width for running mean and std')
  parser.add_argument('--fwd_bits',   type=int, default=8, help='bit-width for layer output')
  args = parser.parse_args()
  return args