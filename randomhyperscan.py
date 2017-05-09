import sys
import numpy as np
from time import time

#from keras.models import Sequential
#from keras.layers import Dense, Activation, Dropout
#from keras.regularizers import l2
#from keras.optimizers import SGD, Adam, Nadam, Adagrad
#from keras.layers.normalization import BatchNormalization
from keras.callbacks import History, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras.backend as K

#own scripts
import misc
from netbuilder import build_Sequential_RH, build_Optimizer_RH
from hyperparameter import RandomHyperPar

#config
result_txt = 'output/result_random_hyperscan.txt'#str(sys.argv[1])
N = int(sys.argv[1])
split = 0.7
initial_patience = 20
lr_divisor = 5.0
bestnet = 'output/temp_%s_best.h5' % time() #now this process has a unique temporary best net
histos = []

#pmssm
from Preprocessor import pmssm, chi2, fulldata
data = fulldata()

#learning utilities
modcp = ModelCheckpoint(bestnet, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#main hyperloop
for i in range(N):
    hp = RandomHyperPar()
    print '\niteration=%s\n' % i
    print hp
    #build model according to hp
    model = build_Sequential_RH(hp)
    opt = build_Optimizer_RH(hp)
    #loss=misc.mae_poisson #if we have an error on chi2, use this!
    model.compile(loss='mae', optimizer=opt) #, metrics=[mean_loss_chi2])

    #shuffle data, so we dont learn hyperparameters for a certain validation set
    data.shuffle()
    x = pmssm(data.data[:,:-1], preproc = hp.pp_pmssm, split = split)
    y = chi2(data.data[:,-1], preproc = hp.pp_chi2, params = [100,25], split = split)

    lr=hp.lr
    first_rounds = History()
    #sometimes lr is too high and we get nan or inf or something..
    model.fit(x.train, y.train, validation_data=(x.test,y.test), epochs=1, batch_size=hp.batch, verbose=1, callbacks=[first_rounds])
    if misc.bad_loss(first_rounds):
        print 'quitting because loss too high %s' % first_rounds.history
        result = misc.result_string(hp, y, earlyquit=True)
        misc.append_to_file(result_txt, result)
        continue

    counter = 0 
    while lr > 10**(-6.05):
        counter += 1

        #learning utility
        patience = max(5, int(initial_patience/(counter)**(0.5))) #patience decrease 
        #No more huge gains expected after we have reduced lr several times. we want to save computation time
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1)
        history = History()

        model.fit(x.train, y.train, validation_data=(x.test,y.test), epochs=1000, batch_size=hp.batch, verbose=1, callbacks=[history,early_stopping,modcp])
        model.load_weights(bestnet)
        lr /= lr_divisor
        K.set_value(model.optimizer.lr, lr)
        print '\n\nNEW LEARNING RATE:', lr#, K.get_value(model.optimizer.lr), model.optimizer.get_config()['lr'], '\n\n'
        histos.append(history)
        if misc.quit_early(histos):
            break #result of run will get saved below!

    print 'final evaluation'
    model.load_weights(bestnet)
    y.evaluation(x, model) #is needed for getting mean_errs 
    result = misc.result_string(hp, y)
    misc.append_to_file(result_txt, result)

