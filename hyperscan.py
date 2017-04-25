import sys
import numpy as np
from numpy.random import RandomState
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Nadam, Adagrad
#from keras.layers.normalization import BatchNormalization
from keras.callbacks import History, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


from netbuilder import build_Sequential
from hyperparameter import Hyperparameter

#pmssm
from Preprocessor import pmssm, chi2, fulldata
data = fulldata()

#some configuration for use of data
split = 9.0/10


data.shuffle()
x = pmssm(data.data[:,:-1], preproc = ['log_norm','min_max'], split = split)
y = chi2(data.data[:,-1], preproc = ['square_cut','div_max'], params = [100,25], split= split)


import keras.backend as K
def mean_loss_chi2(y_true, y_pred):#TODO
    return 100*K.mean(K.abs(y_pred-y_true))





early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

modcp = ModelCheckpoint("best_net_r_chi2.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = History()

#main hyperloop
for i in range

learnrate=10**(-3.0)
while learnrate > 10**(-5.1):
	opt = Nadam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0)
	model.compile(loss='mse', optimizer=opt, metrics=[mean_loss_chi2])
	model.fit(x.train, y.train, validation_data=(x.test,y.test), epochs=400, batch_size=1000, verbose=1, callbacks=[history,early_stopping,modcp])
	model.load_weights('best_net_r_chi2.hdf5')
	learnrate /= 4
	print 'learnrate:', learnrate

