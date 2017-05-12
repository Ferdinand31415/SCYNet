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
N = 3#int(sys.argv[1])
split = 0.7
initial_patience = 4
patience_dec = 0.5 #decreases patience in some way
lr_divisor = 5.0 #we divide each lr epoch by this number
resultfolder = os.environ['HOME']+'/resultSCYNet'
bestnet = resultfolder + '/temp/%s_best.h5' % time() #instance of this script has a unique temporary best net
result_txt = resultfolder + '/randhyperscan_initpat_%s_lrdivisor_%s_patdec_%s_split_%s' % \
            (initial_patience, lr_divisor, patience_dec, split)
if not os.path.isfile(result_txt):
    os.mknod(result_txt)

#results = load_result_file(result_txt)
#best = get_global_best(results)
#del results



################
#main hyperloop#
################
histos = []
try:
    for i in range(N):
        hp = RandomHyperPar(fasttrain=0)
        print '\nhyperparameter number %s' % i
        print hp
        #build model according to hp
        model = build_Sequential_RH(hp)
        opt = build_Optimizer_RH(hp)
        #loss=misc.mae_poisson #if we finally have an error on chi2, use this!
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
        best = 100
        best_in_lr_epoch = []
        while lr_epoch <= 6:
            print '\n\nLEARNING RATE: %s, adjusted lr %s times' % (lr,lr_epoch)#, K.get_value(model.optimizer.lr), model.optimizer.get_config()['lr'], '\n\n'
            lr_epoch += 1
            
            #learning utility
            patience = max(2, int(initial_patience/(lr_epoch)**(patience_dec))) #patience decrease 
            #No more huge gains expected after we have reduced lr several times. we want to save computation time
            
            history = History()
            wait, wait_sudden_catastrophic_loss = 0, 0
            #wait: for checking if we are on 'loss plateau'.
            #wait_sudden_catastrophic_loss: sometimes the algorithm becomes unstable and we get huge losses. reload checkpoint then
            counter = 0
            while wait <= patience:
                hist = model.fit(x.train, y.train, validation_data=(x.test,y.test), epochs=1, batch_size=hp.batch, verbose=1, callbacks=[history,modcp])
                histos.append(history)
                current = hist.history['val_loss'][0]
                if current > 0.6 or np.isnan(current) or np.isinf(current):
                    print 'SUDDEN CATASTROPY %s' % current
                    wait_sudden_catastrophic_loss += 1
                    if wait_sudden_catastrophic_loss > 1:
                        lr_epoch = 100
                        break
                    model.load_weights(bestnet)
                    continue
                if current < best:
                    best = current
                else:
                    wait += 1
                
                #if we have painfully slow convergence
                if lr_epoch == 0 and counter == 4:
                    loss_zero = histos[0].history['val_loss'][0]
                    loss_check = histos[4].history['val_loss'][0]
                    if loss_check < 0.6*loss_zero:
                        lr_epoch = 100
                        break 
                print 'lr_epoch %s, patience %s, current %s, wait %s, best %s\n' % (lr_epoch, patience, current, wait, best)
                counter += 1
            #prepare for new lr_epoch    
            model.load_weights(bestnet)
            lr /= lr_divisor
            K.set_value(model.optimizer.lr, lr)
            
            best_in_lr_epoch.append(best)
            if len(best_in_lr_epoch) > 1:
                loss_last = best_in_lr_epoch[-2]
                loss_current = best_in_lr_epoch[-1]
                if loss_current > 0.95 * loss_last:
                    #improvement this lr epoch is smaller than 5%.
                    #leave lr_epoch while loop. go to final evaluation.
                    break
                
                
            #if misc.quit_early(histos):
            #    break #result of run will get saved below!

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
except KeyboardInterrupt:
    os.remove(bestnet)
    sys.exit() 
