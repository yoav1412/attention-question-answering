import keras.backend as K
from tqdm import tqdm


class ExponentialMovingAverage(object):
    """
    Maintains and exponential moving average on the weights. Used by the EvaluationCallBack.
    """
    def __init__(self, model, decay, weights_list=None, temp_model_pth='temp_model.h5',
                 name='ExponentialMovingAverage'):
        self.model = model
        self.scope_name = name
        self.temp_model_pth = temp_model_pth
        self.decay = decay
        self.averages = {}
        if weights_list is None:
            weights_list = self.model.trainable_weights
        print('EMA: getting weights...')
        for weight in tqdm(weights_list):
            self.averages[weight.name] = K.get_value(weight)

    def average_update(self):
        # run in the end of each batch
        for weight in self.model.trainable_weights:
            prev_val = self.averages[weight.name]
            self.averages[weight.name] = self.decay * prev_val + (1.0 - self.decay) * K.get_value(weight)

    def assign_shadow_weights(self, backup=True):
        if backup:
            self.model.save_weights(self.temp_model_pth)
        print('EMA: assigning weights...')
        for weight in tqdm(self.model.trainable_weights):
            K.set_value(weight, self.averages[weight.name])
        print('\tDone.')
