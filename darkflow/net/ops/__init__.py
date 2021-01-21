from .simple import *
from .convolution import *
from .rnn import *
from .baseop import HEADER, LINE

op_types = {
	'convolutional': convolutional,
	'conv-select': conv_select,
	'connected': connected,
	'maxpool': maxpool,
	'leaky': leaky,
	'relu': relu,  
	'dropout': dropout,
	'flatten': flatten,
	'avgpool': avgpool,
	'softmax': softmax,
	'identity': identity,
	'crop': crop,
	'local': local,
	'select': select,
	'route': route,
	'reorg': reorg,
	'conv-extract': conv_extract,
	'extract': extract,
	'rnn': rnn
}

def op_create(*args):
	layer_type = list(args)[0].type
	print(args)
	return op_types[layer_type](*args)
