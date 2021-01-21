from .layer import Layer
import numpy as np

class rnn_layer(Layer):
    def setup(self, 
              output_size, activation):
        self.activation = activation
        self.inp = input_size
        self.out = output_size
        self.wshape = {
            'biases': [self.out],
            'weights': [self.inp, self.out]
        }
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


    def finalize(self, transpose):
        weights = self.w['weights']
        weights = self.w['weights']
        if weights is None: return
        shp = self.wshape['weights']
        if not transpose:
            weights = weights.reshape(shp[::-1])
            weights = weights.transpose([1,0])
        else: weights = weights.reshape(shp)
        self.w['weights'] = weights
