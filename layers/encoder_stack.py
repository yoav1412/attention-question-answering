from layers.positional_encoding import PositionalEncoding
from layers.layer_normalization import LayerNormalization
from keras.layers import Add, Conv1D, Lambda, Dropout, SeparableConv2D
from conf import Config as cf
import keras.backend as K
from keras.regularizers import l2
from layers.multihead_self_attention import MultiHeadSelfAttention as RefAttention

class SingleEncoderBlock(): # does not inherit from Layer
    def __init__(self, input_dim, output_dim, num_heads, n_convs, kernel_size, pwffn_filters, name, initializer='glorot_uniform', dropout=0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_convs = n_convs
        self.kernel_size = kernel_size
        self.pwffn_filters = pwffn_filters
        self.initializer = initializer
        self.dropout = dropout
        self.layers = []
        self.name = name
        self.num_heads = num_heads
        self._build_block()

    def _build_residual(self, nested_layer, name):
        """
        :param nested_layer: String, representing which layer is to be applied within the residual block ('conv', 'attn', 'pwffn')
        :return:
        """
        layers = []
        layers.append(LayerNormalization())
        assert nested_layer in ['conv', 'attn','pwffn'], "Error: Invalid nested_layer passed to residual block."
        if nested_layer == 'conv':
            layers.extend(self._build_single_conv(name))
        elif nested_layer == 'attn':
            layers.extend(self._build_attn(name))
        elif nested_layer == 'pwffn':
            layers.extend(self._build_pwffn(name))

        if self.dropout > 0:
            layers.append(Dropout(rate=self.dropout))

        layers.append(Add()) # residual add
        return layers

    def _build_single_conv(self, name):
        layers = []
        layers.append(Lambda(lambda x: K.expand_dims(x, axis=2), name=name + "_expandDimsLambda"))
        layers.append(
            SeparableConv2D(
            filters=self.output_dim,
            kernel_size=self.kernel_size,
            padding='same',
            depthwise_initializer=cf.RELU_INIT,
            depthwise_regularizer=l2(cf.L2_LAMBDA),
            activation=cf.CONV_ACTIVATION,
            name=name)
        )
        layers.append(Lambda(lambda x: K.squeeze(x, axis=2), name=name+"_squeezeDimsLambda"))
        return layers

    def _build_attn(self, name):
        layers = []
        additiona_layers = [Conv1D(2 * self.input_dim, 1,
                             kernel_initializer=self.initializer,
                             kernel_regularizer=l2(cf.L2_LAMBDA)),
                            Conv1D(self.input_dim, 1,
                             kernel_initializer=self.initializer,
                             kernel_regularizer=l2(cf.L2_LAMBDA)),
                            RefAttention(self.input_dim, 8, dropout=cf.BETWEEN_LAYER_DROPOUT_RATE, bias=False)]
        layers.extend(additiona_layers)
        return layers

    def _build_pwffn(self, name):
        layers = []
        layers.append(Conv1D(filters=self.pwffn_filters[0], kernel_size=1, activation='relu',
                            name=name + "_1st_pwffn",kernel_regularizer=l2(cf.L2_LAMBDA), kernel_initializer=cf.RELU_INIT))
        layers.append(Conv1D(filters=self.pwffn_filters[1], kernel_size=1, activation='linear',
                             name=name + "_2nd_pwffn", kernel_regularizer=l2(cf.L2_LAMBDA)))
        return layers

    def _build_block(self):
        self.layers.append(PositionalEncoding())
        for i in range(self.n_convs):
            self.layers.extend(self._build_residual(nested_layer='conv', name=self.name+"_conv{}".format(i)))
        self.layers.extend(self._build_residual(nested_layer='attn', name=self.name+'_selfAttn'))
        self.layers.extend(self._build_residual(nested_layer='pwffn', name=self.name+"_pwffn"))

    def _call_residual(self, x, layer_iter, nested_layer, mask=None):
        # save orig input
        orig_input = x
        # apply layer normalization:
        x = next(layer_iter)(x)
        if nested_layer == 'conv':
            x = next(layer_iter)(x)  # 1st Lambda (expand dims)
            x = next(layer_iter)(x)  # apply conv
            x = next(layer_iter)(x)  # 2nd Lambda (squeeze)
        elif nested_layer == 'attn':
            x1 = next(layer_iter)(x)
            x2 = next(layer_iter)(x)
            x = next(layer_iter)([x1, x2, mask])
        elif nested_layer == 'pwffn':
            x = next(layer_iter)(x) # 1st pwffn
            x = next(layer_iter)(x) # 2nd pwffn
        if self.dropout > 0:
            x = next(layer_iter)(x) # apply dropout
        x = next(layer_iter)([orig_input, x]) # Add() (residual connection)
        return x

    def _call_encoder_block(self, x, mask=None):
        layer_iter = iter(self.layers)
        x = next(layer_iter)(x) # positional encoding
        for _ in range(self.n_convs):
            x = self._call_residual(x, layer_iter, nested_layer='conv')
        x = self._call_residual(x, layer_iter, nested_layer='attn', mask=mask)
        x = self._call_residual(x, layer_iter, nested_layer='pwffn')
        return x


class EncoderStack():
    def __init__(self, n_blocks, input_dim, output_dim, num_heads, n_convs, kernel_size, pwffn_filters, name,
                 initializer='glorot_uniform', dropout=0):
        self.n_blocks = n_blocks
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_convs = n_convs
        self.kernel_size = kernel_size
        self.pwffn_filters = pwffn_filters
        self.initializer = initializer
        self.dropout=dropout
        self.blocks = []
        self.name = name
        self.num_heads = num_heads

        self._build_encoder_stack()

    def __call__(self, x, mask=None):
        return self._call_encoder_stack(x, mask)

    def _build_encoder_stack(self):
        for i in range(self.n_blocks):
            block = SingleEncoderBlock(input_dim=self.input_dim, output_dim=self.output_dim, num_heads=self.num_heads,
                                       n_convs=self.n_convs, kernel_size=self.kernel_size, initializer=self.initializer,
                                       pwffn_filters=self.pwffn_filters, name=self.name+"_block{}".format(i),
                                       dropout=self.dropout)
            self.blocks.append(block)

    def _call_encoder_stack(self, x, mask=None):
        for block in self.blocks:
            x = block._call_encoder_block(x, mask)
        return x

