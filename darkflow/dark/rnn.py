from .layer import Layer
import numpy as np

class rnn_layer(Layer):
    def setup(self, input_size,output_size, activation):
        self.activation = activation
        self.inp = input_size
        self.out = output_size
        self.wshape = {
            'biases_1': [self.out],
            'weights_1': [self.inp, self.out],
            'biases_2': [self.out],
            'weights_2': [self.out, self.out],
            'biases_3': [self.out],
            'weights_3': [self.out, self.out] 
        }
        if self.batch_norm:
            self.wshape.update({
                'moving_variance_1'  : [self.out], 
                'moving_mean_1': [self.out], 
                'gamma_1' : [self.out],
                'moving_variance_2'  : [self.out], 
                'moving_mean_2': [self.out], 
                'gamma_2' : [self.out],
                'moving_variance_3'  : [self.out], 
                'moving_mean_3': [self.out], 
                'gamma_3' : [self.out]
            })
            self.h['is_training'] = {
                'feed': True,
                'dfault': False,
                'shape': ()
            }


    def finalize(self, transpose):
        weights_1 = self.w['weights_1']
        weights_2 = self.w['weights_2']
        weights_3 = self.w['weights_3']
        if weights_1 is None: return
        shp = self.wshape['weights_1']
        if not transpose:
            weights_1 = weights_1.reshape(shp[::-1])
            weights_1 = weights_1.transpose([1,0])
        else: weights_1 = weights_1.reshape(shp)
        self.w['weights_1'] = weights_1
        
        if weights_2 is None: return
        shp = self.wshape['weights_2']
        if not transpose:
            weights_2 = weights_2.reshape(shp[::-1])
            weights_2 = weights_2.transpose([1,0])
        else: weights_2 = weights_2.reshape(shp)
        self.w['weights_2'] = weights_2
        
        if weights_3 is None: return
        shp = self.wshape['weights_3']
        if not transpose:
            weights_3 = weights_3.reshape(shp[::-1])
            weights_3 = weights_3.transpose([1,0])
        else: weights_3 = weights_3.reshape(shp)
        self.w['weights_3'] = weights_3
        
        
