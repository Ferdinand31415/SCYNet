import numpy as np
from numpy.random import RandomState
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU
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
x = pmssm(data.data[:,:-1], preproc = ['div_max'], split = split)
y = chi2(data.data[:,-1], preproc = ['square_cut','min_max'], params = [100,25], split= split)

model = Sequential()
LReLu = LeakyReLU(alpha=0.2)
n, act, init = 500, 'linear', 'glorot_uniform'
model.add(Dense(n, kernel_initializer=init,
#		kernel_initializer='zero',
		activation=act,
		input_dim=x.train.shape[1]))
for i in range(3):
    model.add(Dense(n-0*i, kernel_initializer=init,activation=act))#, W_regularizer=l2(0.001)))
    model.add(Dropout(0.08))
    model.add(LReLu)
model.add(Dense(1, kernel_initializer=init,activation='relu'))


net = './output/bestnet.hdf5'
if os.path.isfile(net):
    os.remove(net)

early_stopping = EarlyStopping(monitor='val_loss', patience=400, mode='min', verbose=1)
modcp = ModelCheckpoint(net, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = History()

learnrate=10**(-3.0)
opt = Nadam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0)
model.compile(loss='mae', optimizer=opt)

lr_epoch = 0
while learnrate > 10**(-7.1):
    K.set_value(model.optimizer.lr, learnrate)
    model.fit(x.train, y.train, validation_data=(x.test,y.test), epochs=1000, batch_size=1000, verbose=1, callbacks=[history,early_stopping,modcp])
    model.load_weights(net)
    learnrate /= 4
    lr_epoch += 1
    print 'learnrate:', learnrate, '\tAmount of times lr has beed adjusted:',lr_epoch

y.evaluation(x, model)

