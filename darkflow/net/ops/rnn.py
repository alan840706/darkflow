import tensorflow.contrib.slim as slim
from .baseop import BaseOp
import tensorflow as tf
import numpy as np

class rnn(BaseOp):
    def forward(self):

        self.out = tf.nn.xw_plus_b(
            self.inp.out,
            self.lay.w['weights_1'], 
            self.lay.w['biases_1'], 
            name = self.scope)
	
	if self.lay.batch_norm: 
            temp = self.batchnorm(self.lay, temp)
        
        self.out = tf.nn.xw_plus_b(
            self.inp.out,
            self.lay.w['weights_2'], 
            self.lay.w['biases_2'], 
            name = self.scope)
        
        self.out = tf.nn.xw_plus_b(
            self.inp.out,
            self.lay.w['weights_3'], 
            self.lay.w['biases_3'], 
            name = self.scope)
        
        

    def batchnorm(self, layer, inp):
        if not self.var:
            temp = (inp - layer.w['moving_mean_1'])
            temp /= (np.sqrt(layer.w['moving_variance_1']) + 1e-5)
            temp *= layer.w['gamma_1']
            return temp
        else:
            args = dict({
                'center' : False, 'scale' : True,
                'epsilon': 1e-5, 'scope' : self.scope,
                'updates_collections' : None,
                'is_training': layer.h['is_training'],
                'param_initializers': layer.w
                })
            return slim.batch_norm(inp, **args)
        
    def speak(self):
		layer = self.lay
		args = [layer.inp, layer.out]
		args += [layer.activation]
		msg = 'full {} x {}  {}'
		return msg.format(*args)
