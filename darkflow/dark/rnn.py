from .layer import Layer
import numpy as np

class rnn_layer(Layer):
    def setup(self, ksize, c, n, stride, 
              pad, batch_norm, activation,groups):
        self.batch_norm = bool(batch_norm)
        self.activation = activation
        self.stride = stride
        self.ksize = ksize
        self.pad = pad
        self.groups = groups
        if groups>1:
            self.dnshape = [1,c , ksize, ksize] # darknet shape
            self.wshape = dict({
                'biases': [n], 
                'kernel': [ksize, ksize, c, 1]
            })
        else:   
            self.dnshape = [n,c , ksize, ksize] # darknet shape
            self.wshape = dict({
                'biases': [n], 
                'kernel': [ksize, ksize, c, n]
            })
        if self.batch_norm:
            self.wshape.update({
                'moving_variance'  : [n], 
                'moving_mean': [n], 
                'gamma' : [n]
            })
            self.h['is_training'] = {
                'feed': True,
                'dfault': False,
                'shape': ()
            }

    def finalize(self, _):
        """deal with darknet"""
        kernel = self.w['kernel']
        if kernel is None: return
        kernel = kernel.reshape(self.dnshape)
        kernel = kernel.transpose([2,3,1,0])
        self.w['kernel'] = kernel
