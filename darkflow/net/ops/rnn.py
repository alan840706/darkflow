import tensorflow.contrib.slim as slim
from .baseop import BaseOp
import tensorflow as tf
import numpy as np

class rnn(BaseOp):
    def forward(self):
        print(self.inp.out)
        self.out = tf.nn.xw_plus_b(
            self.inp.out,
            self.lay.w['weights_1'], 
            self.lay.w['biases_1'], 
            name = self.scope)

    def batchnorm(self, layer, inp):
        if not self.var:
            temp = (inp - layer.w['moving_mean'])
            temp /= (np.sqrt(layer.w['moving_variance']) + 1e-5)
            temp *= layer.w['gamma']
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
        l = self.lay
        args = [l.ksize] * 2 + [l.pad] + [l.stride]
        args += [l.batch_norm * '+bnorm']
        args += [l.activation]
        msg = 'conv {}x{}p{}_{}  {}  {}'.format(*args)
        return msg
