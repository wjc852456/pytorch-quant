# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
from torch.autograd import Variable
import math
import quant
input = Variable(torch.FloatTensor([ [12.3000],[23.5000],[129.2000],[-293.1000] ]) )
bits = 12
# linear_quantize
sf = quant.compute_integral_part(input,0.1)
sf_linear = 12-1-sf
linear_q_res = quant.linear_quantize(input,sf_linear,bits)

# min_max_quantize
minmax_q_res = quant.min_max_quantize(input,bits)
#print "minmax_q_res\n",minmax_q_res
# log_minmax_quantize
#log_q_res = quant.log_minmax_quantize(input,bits)
#print log_q_res

