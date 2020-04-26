from keras.layers import Lambda, Dropout, Conv1D
from keras.regularizers import l2
from conf import Config as cf

class HighwayNetwork():
    def __init__(self, filters, n_layers=2, initializer='glorot_uniform', regularizer=l2(cf.L2_LAMBDA), dropout=0):
        self.layers = []
        self.initializer = initializer
        self.regularizer = regularizer
        self.n_layers = n_layers
        self.dropout = dropout

        self._build(filters)


    def _build(self, filters):
        # Projection layer:
        self.layers.append(Conv1D(filters, kernel_size=1, kernel_initializer=self.initializer,
                                  kernel_regularizer=self.regularizer,name='highway_projection'))
        # Highway Layers:
        for i in range(self.n_layers):
            self.layers.append(Conv1D(filters, 1,
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regularizer,
                                     activation='sigmoid',
                                     name='highway' + str(i) + '_gate'))
            self.layers.append(Conv1D(filters, 1,
                                      kernel_initializer=self.initializer,
                                      kernel_regularizer=self.regularizer,
                                      activation='linear',
                                      name='highway' + str(i) + '_linear'))

    def __call__(self, x):
        # apply projection:
        x = self.layers[0](x)
        # apply highway layers:
        for i in range(self.n_layers):
            T = self.layers[i * 2 + 1](x)
            H = self.layers[i * 2 + 2](x)
            H = Dropout(self.dropout)(H)
            x = Lambda(lambda v: v[0] * v[1] + v[2] * (1 - v[1]))([H, T, x])
        return x