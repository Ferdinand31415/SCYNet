import sys
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Nadam, Adagrad
#from keras.layers.normalization import BatchNormalization
from keras.callbacks import History, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras.backend as K

#own scripts
import misc
from netbuilder import build_Sequential, build_Optimizer
from hyperparameter import Hyperparameter

#config
result_txt = str(sys.argv[1])
N = int(sys.argv[2])
split = 0.8
patience = 2
histos = []

#pmssm
from Preprocessor import pmssm, chi2, fulldata
data = fulldata()

#learning utilities
modcp = ModelCheckpoint("bestnet.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#main hyperloop
for i in range(N):
    hp = RandomHyperPar()
    print hp
    #build model according to hp
    model = build_Sequential_RH(hp)
    opt = build_Optimizer_RH(hp)
    model.compile(loss='mae', optimizer=opt) #, metrics=[mean_loss_chi2])

    #shuffle data, so we dont learn hyperparameters for a certain validation set
    data.shuffle()
    x = pmssm(data.data[:,:-1], preproc = hp.pp_pmssm, split = split)
    y = chi2(data.data[:,-1], preproc = hp.pp_chi2, params = [100,25], split = split)

    #find starting learing rate. Not too high(instable + nonconverging)
    #or too low (too slow learning)

    try:
        lr=hp.lr
        first_rounds = History()
        #sometimes lr is too high and we get nan or inf or something..
        model.fit(x.train, y.train, validation_data=(x.test,y.test), epochs=1, batch_size=hp.batch, verbose=1, callbacks=[first_rounds])
        if misc.bad_loss(first_rounds):
            print 'quitting because loss too high %s' % first_rounds.history
            continue

        counter = 0 
        while lr > 10**(-7.1):
            #learning utility
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1)
            patience = int(patience/1.15) #patience decrease. No more huge gains expected after we have reduced lr. we want to save computation time

            history = History()
            model.fit(x.train, y.train, validation_data=(x.test,y.test), epochs=400, batch_size=hp.batch, verbose=1, callbacks=[history,early_stopping,modcp])
            model.load_weights('bestnet.hdf5')
            lr /= 5.0
            K.set_value(model.optimizer.lr, lr)
            print '\n\nNEW LEARNING RATE:', lr, K.get_value(model.optimizer.lr), model.optimizer.get_config()['lr'], '\n\n'
            histos.append(history)
            if misc.quit_early(histos):
                break
    except KeyboardInterrupt:
        #put point into waiting again
        sys.exit('manually canceled because of keyboard interrupt')

    print 'final evaluation'
    model.load_weights('bestnet.hdf5')
    y.evaluation(x, model) #is needed for getting mean_errs 
    result = hp.string() + 'error:' + str(y.mean_errors['0.0-100.0'])+'\n'
    misc.add_to_file(result_txt, result)

