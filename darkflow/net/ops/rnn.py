class rnn(BaseOp):
    def forward(self):
        pad = [[self.lay.pad, self.lay.pad]] * 2;
        #print([[0, 0]] + pad + [[0, 0]])
        #temp = tf.pad(self.inp.out, [[0, 0]] + pad + [[0, 0]])
        temp = tf.space_to_batch_nd(self.inp.out, [1,1], pad, name=None)  
        if self.lay.groups == 1:
            temp = tf.nn.conv2d(temp, self.lay.w['kernel'], padding = 'VALID', 
                name = self.scope, strides = [1] + [self.lay.stride] * 2 + [1])
        else:
            temp = tf.nn.depthwise_conv2d(temp, self.lay.w['kernel'] ,padding = 'VALID',
                strides = [1] + [self.lay.stride] * 2 + [1])
        if self.lay.batch_norm: 
            temp = self.batchnorm(self.lay, temp)
        self.out = tf.nn.bias_add(temp, self.lay.w['biases'])

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
