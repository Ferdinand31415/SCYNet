#general
import sys, os

if os.environ['LOGNAME'] == 'feiteneuer':#aachen desktop
    stop_time = 3600
elif os.environ['LOGNAME'] == 'fe918130':#aachen cluster
    stop_time = 30
elif os.environ['LOGNAME'] == 'eiteneuer':#baf cluster
    stop_time = 300
import numpy as np
import time


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
N = 100#int(sys.argv[1])
split = 0.8
initial_patience = 65
patience_dec = 0.5#decreases patience . patience -> initial_patience / (lr_epoch)**patience_dec
lr_divisor = 4.0 #we divide each lr epoch by this number
resultfolder = os.environ['HOME']+'/resultSCYNet'
bestnet = resultfolder + '/temp/%s_best.h5' % time.time() #instance of this script has a unique temporary best net
result_txt = resultfolder + '/result_initpat_%s_patdec_%s_lrdec_%s_spl_%s' % \
            (initial_patience, lr_divisor, patience_dec, split)
if not os.path.isfile(result_txt):
    os.mknod(result_txt)

#results = load_result_file(result_txt)
#best = get_global_best(results)
#del results
#TODO, maybe save some global best model?



################
#main hyperloop#
################
try:
    for i in range(N):
        histos = []
        timecheck_occured = False
        stopped = False
        hp = RandomHyperPar(fasttrain=False)
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

        modcp = ModelCheckpoint(bestnet, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

        start_time = time.time()
        lr=hp.lr
        lr_epoch = 0
        epoch = 0
        best = 100
        best_in_lr_epoch = []
        while ((lr_epoch <= 10) and not stopped):
            print '\n\nLEARNING RATE: %s, adjusted lr %s times' % (lr,lr_epoch)#, K.get_value(model.optimizer.lr), model.optimizer.get_config()['lr'], '\n\n'
            lr_epoch += 1
            
            #learning utility
            patience = max(2, int(initial_patience/(lr_epoch)**(patience_dec))) #patience decrease 
            #No more huge gains expected after we have reduced lr several times. we want to save computation time
            
            history = History()
            wait, wait_sudden_catastrophic_loss = 0, 0
            #wait: for checking if we are on 'loss plateau'.
            #wait_sudden_catastrophic_loss: sometimes the algorithm becomes unstable and we get huge losses. reload checkpoint then
            while wait <= patience:
                hist = model.fit(x.train, y.train, validation_data=(x.test,y.test), epochs=1, batch_size=hp.batch, verbose=1, callbacks=[history,modcp])

                histos.append(history)
                current = hist.history['val_loss'][0]

                #check after first epoch if the loss is too damn high
                if epoch == 0 and lr_epoch == 1 and current > 0.35:
                    print 'FIRST LOSS IS TOO DAMN HIGH, loss: %s' % current
                    stopped = True
                    N -= 1
                    break

                #check if optimizer goes berserk
                if current > 0.6 or np.isnan(current) or np.isinf(current):
                    print 'SUDDEN CATASTROPY %s' % current
                    wait_sudden_catastrophic_loss += 1
                    if wait_sudden_catastrophic_loss > 2:
                        stopped = True
                        break
                    model.load_weights(bestnet)
                    continue
                
                #early stopping, check if we hit plateau for val_loss
                if current < best:
                    if current > 0.99 * best:
                        patience = max(2, patience - 1)
                    best = current
                    wait = 0
                else:
                    wait += 1
                
                #if we have painfully slow convergence
                if epoch == 4:
                    loss_zero = histos[0].history['val_loss'][0]
                    loss_check = histos[4].history['val_loss'][0]
                    if loss_check < 0.90*loss_zero:
                        print 'normal training stop, convergence reached, loss_zero:%s, loss_check:%s' % (loss_zero, loss_check)
                        stopped = True
                        N -= 1
                        break

                #check if hp is a good realistic candidate by comparing with some rough benchmark
                #print '****', not timecheck_occured, time.time() - start_time > stop_time
                if not timecheck_occured and (time.time() - start_time > stop_time) and epoch > 2:
                    timecheck_occured = True
                    if current > 0.09:
                        print 'TIMEOUT, LOSS IS %s, SAD!, YOU NEED TO MAKE LOSS small AGAIN' % current
                        stopped = True
                        N -= 1
                        break
                        
                #end of epoch
                print 'lr_epoch %s, patience %s, current %s, wait %s, best %s\n' % (lr_epoch, patience, current, wait, best)
                epoch += 1

            #prepare for new lr_epoch    
            model.load_weights(bestnet)
            lr /= lr_divisor
            K.set_value(model.optimizer.lr, lr)
            
            best_in_lr_epoch.append(best)
            
            #check if almost no learing across lr_epochs
            if len(best_in_lr_epoch) > 1:
                loss_last = best_in_lr_epoch[-2]
                loss_current = best_in_lr_epoch[-1]
                if loss_current > 0.95 * loss_last:
                    print 'ALMOST NO LEARNING ACROSS lr_epochs, best loss last lr_epoch:%s, best loss current epoch%s' % (loss_last, loss_current)
                    #improvement this lr epoch is smaller than 5%.
                    #leave lr_epoch while loop. go to final evaluation.
                    break
                
                
            #if misc.quit_early(histos):
            #    break #result of run will get saved below!

        if not stopped:
            print 'final evaluation'
            model.load_weights(bestnet)
            y.evaluation(x, model) #is needed for getting mean_errs
            for value in y.mean_errors.values():
                if np.isnan(value[0]) or np.isnan(value[1]):
                    print 'WARNING: got nan %s' % value
                    continue
            if y.err < 1.3:
                misc.savemod(model, x, y)
        result = misc.result_string(hp, x.back_info, y, initial_patience, earlyquit = stopped)
        misc.append_to_file(result_txt, result)
        #clean up
        try:
            os.remove(bestnet)
        except:
            print 'ERROR in removing model/bestnet'
            #sys.exit()
        del model
        del x
        del y
except KeyboardInterrupt:
    try:
        os.remove(bestnet)
    except OSError:
        pass
    sys.exit()
