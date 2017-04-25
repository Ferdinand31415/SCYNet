import sys
import numpy as np
from numpy.random import RandomState
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Nadam, Adagrad
#from keras.layers.normalization import BatchNormalization
from keras.callbacks import History, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


from netbuilder import build_Sequential, build_Optimizer
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

#main hyperloop
if len(sys.argv) == 4:
    paramfile, lower, upper = sys.argv[1:4]
else:
    lower, upper = 0, 0


paramfile = 'paramfile.pkl'
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.667, patience=5, min_lr=10**(-4.6), verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
modcp = ModelCheckpoint("bestnet.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

histos = []

h = Hyperparameter(paramfile=paramfile)
for i in range(lower, upper + 1):
    #h.running(i) #signal to all other jobs that this id is training now.
    h.print_par(i)
    model = build_Sequential(h.p[i])
    opt = build_Optimizer(h.p[i])
    
    lr=10**(-3.0)
    while lr > 10**(-5.1):
        history = History()
        model.compile(loss='mse', optimizer=opt, metrics=[mean_loss_chi2])
        model.fit(x.train, y.train, validation_data=(x.test,y.test), epochs=400, batch_size=1000, verbose=1, callbacks=[history,early_stopping,modcp])
        model.load_weights('bestnet.hdf5')
        lr /= 10.0
        print 'lr:', lr
        histos.append(history)
        if len(histos) > 1:
            h_old = histos[-2].history['val_loss']
            h_new=  histos[-1].history['val_loss']
            if min(h_old) < min(h_old): #if true, no improvement at all with smaller lr, we can abort learning for time reasons
               break
    model.load_weights('bestnet.hdf5')
    y.evaluation(x, model) #is needed for getting mean_errs
    h.finished(i, y.mean_errors)
                  


# model.fit(x.train,y.train, validation_data =(x.test, y.test), epochs = n,batch_size=1000, callbacks = [reduce_lr, modcp, history])


y.evaluation(x, model)
