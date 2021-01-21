from .layer import Layer
import numpy as np

class rnn_layer(Layer):
    def setup(self, input_size, 
              output_size, activation):
        self.activation = activation
        self.inp = input_size
        self.out = output_size
        self.wshape = {
            'biases': [self.out],
            'weights': [self.inp, self.out]
        }

    def finalize(self, transpose):
        weights = self.w['weights']
        if weights is None: return
        shp = self.wshape['weights']
        if not transpose:
            weights = weights.reshape(shp[::-1])
            weights = weights.transpose([1,0])
        else: weights = weights.reshape(shp)
        self.w['weights'] = weights
