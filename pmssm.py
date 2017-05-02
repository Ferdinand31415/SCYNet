import numpy as np
from numpy.random import RandomState
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Nadam, Adagrad
#from keras.layers.normalization import BatchNormalization
from keras.callbacks import History, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import sys
import keras.backend as K

from copy import deepcopy
from Preprocessor import pmssm, chi2, shuffle_data, fulldata

data = fulldata()

split = 9.0/10
use_only = range(11)

data.shuffle()
x = pmssm(data.data[:,:-1], preproc = ['log_norm','min_max'], split = split)
y = chi2(data.data[:,-1], preproc = ['square_cut','div_max'], params = [100,25], split= split)

model = Sequential()
LReLu = keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
n, act, init = 300, LReLu, 'glorot_uniform'
model.add(Dense(n, kernel_initializer=init,
#		kernel_initializer='zero',
		activation=act,
		input_dim=x.train.shape[1]))
for i in range(2):
    model.add(Dense(n-0*i, kernel_initializer=init,activation=act))#, W_regularizer=l2(0.001)))
    model.add(Dropout(0.08))
model.add(Dense(1, kernel_initializer=init,activation=act))



early_stopping = EarlyStopping(monitor='val_loss', patience=18, mode='min', verbose=1)
#from misc import troll
#early_stopping = troll(patience=4) 
modcp = ModelCheckpoint("bestnet.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = History()

learnrate=10**(-3)
opt = Nadam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0)
model.compile(loss='mae', optimizer=opt)

lr_epoch = 0
while learnrate > 10**(-7.1):
    K.set_value(model.optimizer.lr, learnrate)
    model.fit(x.train, y.train, validation_data=(x.test,y.test), epochs=1000, batch_size=1000, verbose=1, callbacks=[history,early_stopping,modcp])
    model.load_weights('bestnet.hdf5')
    learnrate /= 4
    lr_epoch += 1
    print 'learnrate:', learnrate, '\tAmount of times lr has beed adjusted:',lr_epoch

y.evaluation(x, model)

