#from .builder import build_unet
#from ..utils import freeze_model
#from ..utils import legacy_support
#from ..backbones import get_backbone, get_feature_layers
from keras.layers import Conv2D,BatchNormalization, Input, Add,concatenate
#from keras.layers.merge import Concatenate
from keras.models import Sequential,Model 
import numpy as np 

old_args_map = {
    'freeze_encoder': 'encoder_freeze',
    'skip_connections': 'encoder_features',
    'upsample_rates': None,  # removed
    'input_tensor': None,  # removed
}




#
#def shallow_CNN(num_bands = None, k_1 = None, k_2 = None, k_3 = None):
#    """ A shallow CNN is a fully convolution neural network  """
#    active = 'relu'
#    active2 = 'linear'
#    inp = Input(shape=(None, None, num_bands))
##    bn = BatchNormalization()(inp)
#    l1 = Conv2D(64, kernel_size=k_1, activation= active, padding='same', kernel_initializer='he_normal' )(inp)
#    l2 = Conv2D(48, kernel_size=k_2, activation=active, padding='same', kernel_initializer='he_normal')(l1)
#    l3 = Conv2D(32, kernel_size=k_3, activation=active, padding='same', kernel_initializer='he_normal')(l2)
#    l4 = Conv2D(1, kernel_size=k_3, activation=active2, padding='same', kernel_initializer='he_normal',name="details")(l3)
##    l4= Conv2D(1, kernel_size=k_3, activation=active2, padding='same', kernel_initializer='he_normal')(l3)
##    inp2 = Input(shape=(None, None, 1))
#    inp1 = Input(shape=(None, None, 1))
#    out = Add(name="band")([l4, inp1])
#    model = Model([inp, inp1], [l4,out], name='shallow_CNN')
#    
##    out= Conv2D(1, kernel_size=k_3, activation='relu', padding='same', kernel_initializer='he_normal',name="nothing")(out1)
##    model = Model(inp, l4, name='shallow_CNN')
#    return model

def shallow(num_bands = None, k_1 = None, k_2 = None, k_3 = None):
    """ A shallow CNN is a fully convolution neural network  """
    active = 'relu'
    active3 = 'softmax'
    inp = Input(shape=(None, None, num_bands))
    bn = BatchNormalization()(inp)
    l1 = Conv2D(64, kernel_size=k_1, activation= active, padding='same', kernel_initializer='he_normal' )(bn)

    l2 = Conv2D(48, kernel_size=k_2, activation=active, padding='same', kernel_initializer='he_normal')(l1)
    l3 = Conv2D(32, kernel_size=k_3, activation=active, padding='same', kernel_initializer='he_normal')(l2)
    out = Conv2D(1, kernel_size=k_3, activation=active3, padding='same', kernel_initializer='he_normal')(l3)
    
#    out= Conv2D(1, kernel_size=k_3, activation='relu', padding='same', kernel_initializer='he_normal',name="nothing")(out1)
    model = Model(inp, out, name='shallow')
    return model

