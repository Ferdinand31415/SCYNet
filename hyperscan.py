import numpy as np
from numpy.random import RandomState
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Nadam, Adagrad
#from keras.layers.normalization import BatchNormalization
from keras.callbacks import History, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


from copy import deepcopy
path ='13TeV_chi2_disjoint_2'

from Preprocessor import pmssm, chi2, shuffle_data
split = 9.0/10
x = pmssm(preproc = ['log_norm','min_max'], split = split)
y = chi2(['square_cut','div_max'], [100,25], split= split)
#shuffle_data(x, y)


import keras.backend as K
def mean_loss_chi2(y_true, y_pred):
    return 100*K.mean(K.abs(y_pred-y_true))


