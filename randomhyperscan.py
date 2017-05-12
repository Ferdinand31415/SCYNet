#general
import sys, os
import numpy as np
from time import time

#Keras
from keras.callbacks import History, ModelCheckpoint, EarlyStopping
import keras.backend as K

#own scripts
import misc
from netbuilder import build_Sequential_RH, build_Optimizer_RH
from hyperparameter import RandomHyperPar
from Preprocessor import pmssm, chi2, fulldata

####################
#config the hp scan#
####################
N = 10000#int(sys.argv[1])
split = 0.7
initial_patience = 2
patience_dec = 1.2
lr_divisor = 20.0
resultfolder = os.onviron['HOME']+'/resultSCYNet'
bestnet = resultfolder + '/temp/%s_best.h5' % time() #instance of this script has a unique temporary best net
result_txt = resultfolder + '/randhyperscan_initpat_%s_lrdivisor_%s_patdec_%s_split_%s' % \
            (initial_patience, lr_divisor, patience_dec, split)
if not os.path.isfile(result_txt):
    os.mknod(result_txt)

#results = load_result_file(result_txt)
#best = get_global_best(results)
#del results

histos = []


################
#main hyperloop#
################
for i in range(N):
    hp = RandomHyperPar(fasttrain=True)
    print '\niteration=%s\n' % i
    print hp
    #build model according to hp
    model = build_Sequential_RH(hp)
    opt = build_Optimizer_RH(hp)
    #loss=misc.mae_poisson #if we have an error on chi2, use this!
    model.compile(loss='mae', optimizer=opt) #, metrics=[mean_loss_chi2])

    #shuffle data, so we dont learn hyperparameters for a certain validation set
    data = fulldata()
    data.shuffle()
    x = pmssm(data.data[:,:-1], preproc = hp.pp_pmssm, split = split)
    y = chi2(data.data[:,-1], preproc = hp.pp_chi2, params = [100,25], split = split,verbose=False)

    modcp = ModelCheckpoint(bestnet, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    lr=hp.lr
    first_rounds = History()
    #sometimes lr is too high and we get nan or inf or something..
    model.fit(x.train, y.train, validation_data=(x.test,y.test), epochs=1, batch_size=hp.batch, verbose=1, callbacks=[first_rounds])
    if misc.bad_loss(first_rounds):
        print 'quitting because loss too high %s' % first_rounds.history
        result = misc.result_string(hp, x.back_info, y, initial_patience, earlyquit=True)
        misc.append_to_file(result_txt, result)
        continue

    lr_epoch = 0 
    while lr_epoch <= 6
        print '\n\nLEARNING RATE: %s, adjusted lr %s times' % (lr,lr_epoch)#, K.get_value(model.optimizer.lr), model.optimizer.get_config()['lr'], '\n\n'
        lr_epoch += 1

        #learning utility
        patience = max(2, int(initial_patience/(lr_epoch)**(patience_dec))) #patience decrease 
        #No more huge gains expected after we have reduced lr several times. we want to save computation time
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1)
        history = History()

        model.fit(x.train, y.train, validation_data=(x.test,y.test), epochs=1000, batch_size=hp.batch, verbose=1, callbacks=[history,early_stopping,modcp])
        model.load_weights(bestnet)
        lr /= lr_divisor
        K.set_value(model.optimizer.lr, lr)
        histos.append(history)
        if misc.quit_early(histos):
            break #result of run will get saved below!

    print 'final evaluation'
    model.load_weights(bestnet)
    y.evaluation(x, model) #is needed for getting mean_errs
    for value in y.mean_errors.values():
        if np.isnan(value[0]) or np.isnan(value[1]):
            print 'WARNING: got nan %s' % y.mean_errors
            sys.exit()
            continue
    result = misc.result_string(hp, x.back_info, y, initial_patience)
    misc.append_to_file(result_txt, result)
    #clean up
    try:
        os.remove(bestnet)
    except:
        print 'ERROR in removing model/bestnet'
        sys.exit()
    del model
    del x
    del y
 
