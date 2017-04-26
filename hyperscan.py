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
split = 0.8
patience = 2

def almost_no_improvement(histos):
    '''returns true if almost no improvement at all even though
    we had smaller lr. We can then abort learning for time reasons
    in this case'''
    if len(histos) < 2:
        return False
    h_old = histos[-2].history['val_loss']
    h_new = histos[-1].history['val_loss']
    if min(h_old) / min(h_old) < 0.005:
        return True
    else:
        return False



import keras.backend as K
def mean_loss_chi2(y_true, y_pred):#TODO
    return 100*K.mean(K.abs(y_pred-y_true))

early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
modcp = ModelCheckpoint("best_net_r_chi2.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

if len(sys.argv) == 4:
    paramfile = sys.argv[1]
    start_idx, end_idx = int(sys.argv[2]), int(sys.argv[3])
else:
    print 'WARNING: i dont use params from sys.argv()'
    start_idx, end_idx = 0, 10
    paramfile = 'paramfile.pkl'


early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1)
modcp = ModelCheckpoint("bestnet.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

histos = []

h = Hyperparameter(paramfile=paramfile)

#main hyperloop
for id in range(start_idx, end_idx + 1):
    if h.p[id]['status'] != 'waiting':
        #if we are here, hp is running or finished
        print 'hp[%s] is %s, skip' % (id, h.p[id]['status'])
        continue
    
    h.running(id) #signal to all other possible jobs that this id is training now.
    print h
    #h.print_me()
    h.print_par(id)
    model = build_Sequential(h.p[id])
    opt = build_Optimizer(h.p[id])
    
    #shufffle data, so we dont learn hyperparameters for a certain validation set
    data.shuffle()
    x = pmssm(data.data[:,:-1], preproc = ['log_norm','min_max'], split = split)
    y = chi2(data.data[:,-1], preproc = ['square_cut','div_max'], params = [100,25], split= split)

    try:
        lr=10**(-3.0)
        while lr > 10**(-5.1):
            history = History()
            model.compile(loss='mae', optimizer=opt, metrics=[mean_loss_chi2])
            model.fit(x.train, y.train, validation_data=(x.test,y.test), epochs=400, batch_size=1000, verbose=1, callbacks=[history,early_stopping,modcp])
            model.load_weights('bestnet.hdf5')
            lr /= 10.0
            print 'lr:', lr
            histos.append(history)
            if almost_no_improvement(histos):       
                break
    except KeyboardInterrupt:
        #put point into waiting again
        h.waiting(id)
        sys.exit('manually canceled because of keyboard interrupt')

    print 'final evaluation'
    model.load_weights('bestnet.hdf5')
    y.evaluation(x, model) #is needed for getting mean_errs
    h.finished(id, y.mean_errors, model)
                  

