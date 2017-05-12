import numpy as np
import time
import os

def append_to_file(path, result, lockfile='lock'):
    '''has inbuilt protection agains 
    several processes writing to it.
    But make sure all processes know the same lockfile'''
    while True:
        try:
            os.mknod(lockfile)
        except(OSError):
            time.sleep(0.05)
            continue
        f = open(str(path), 'a')
        f.write(result)
        f.close()
        os.remove(lockfile)
        break

def smart_start(x, y, model, batch):
    '''unfinished'''
    try_out = np.logspace(-1.5,-3.5,10)
    histos = []
    history = History()
    for lr in try_out:
        K.set_value(model.optimizer.lr, lr)
        model.fit(x.train, y.train, validation_data=(x.test,y.test),\
            epochs=3, batch_size=batch, verbose=1,\
            callbacks=[history,early_stopping,modcp])

def bad_loss(first_rounds, maxloss=0.4):
    '''test if bad loss is present, abort training if true'''
    loss = first_rounds.history['val_loss'][0]
    if loss > maxloss or np.isnan(loss) or np.isinf(loss):
        return True
    else:
        return False

def bad_loss_anywhere(histos, maxloss=0.6):
    '''test if badloss is present anywhere in history, abort if true'''
    if len(histos) < 2:
        return False
    history = histos[-1].history['val_loss']
    if max(history) > maxloss:
        return True
    else:
        return False

def quit_early(histos):
    '''at the moment checks only for "almost no improvement" '''
    return almost_no_improvement(histos)\
        or bad_loss_anywhere(histos)

def result_string(hp, xback_info, y, initialpatience, earlyquit=False):
    '''save this to a txt file. its the result of the hyperrandomscan'''
    res = hp.string()
    res += 'initialpatience?'+str(initialpatience)+';'
    res += 'chi2trafo?'+str(y.back_info)+';'
    res += 'pmssmtrafo?'+str(xback_info)+';'
    if earlyquit:
        res += 'error?10.0'
    else:
        res += 'error?'+str(y.mean_errors['0.0-100.0'][1])
    return res + '\n'

def load_result_file(path):
    data = []
    with open(path,'r') as file:
        lines = file.readlines()
    for l in lines:
        data.append(result_string_to_dict(l,verbose=False))
    return data

def get_global_best(data):
    best = 100
    if len(data) == 0:
        return best
    for d in data:
        err = get_error(d['error'])
        if err < best:
            best = err
    return best

def savemod(model, x, y):
    err = y.mean_errors['0.0-100.0'][1]
    directory = os.environ['HOME'] + '/resultSCYNet/nets/%s' % err
    if not os.path.exists(directory):
        os.makedirs(directory)
    model.save(directory + '/model_%s.h5' % err)
    with open(directory + '/model_%s.txt' % err,'a') as f:
        f.write(str(x.back_info)+'\n')
        f.write(str(y.back_info)+'\n')

def result_string_to_dict(line,verbose=True):
    hp = {}
    L = line.split(';')
    for l in L:
        key, value = l.split('?')
        if verbose: print key, '\t', value
        hp.update({str(key):str(value)})
        try:
            hp[key] = eval(hp[key])
        except NameError:
            pass
    return hp

def load_result_file(path):
    data = []
    with open(path,'r') as file:
        lines = file.readlines()
    for l in lines:
        data.append(result_string_to_dict(l,verbose=False))
    return data


def almost_no_improvement(histos, improvefrac = 0.04):
    '''returns true if almost no improvement at all even though
    we had smaller lr. We can then abort learning for time reasons
    in this case'''
    if len(histos) < 2:
        return False
    h_old = histos[-2].history['val_loss']
    h_new = histos[-1].history['val_loss']
    if min(h_new) / min(h_old) < improvefrac:
        return True
    else:
        return False

import keras.backend as K
def mean_loss_chi2(y_true, y_pred):
    return 100*K.mean(K.abs(y_pred-y_true))


def mae_poisson(stat_error):
    def loss(y_true, y_pred):
        return K.mean(K.abs(y_pred-ytrue)/stat_error) #just something random to point out how it works
    return loss

from keras.callbacks import EarlyStopping
class CustomEarlyStopper(EarlyStopping):
    '''was intended to stop training if loss is too high. but i dont use it'''
    def __init__(self, patience=4):
        super(CustomEarlyStopper, self).__init__()
        self.patience=patience

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if epoch == 3 and logs['loss'] > 0.75:
            self.model.stop_training = True
            print('\nWARNING: HARD STOP BECAUSE OF TOO HIGH LOSS %s!' % logs['loss'])
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0 
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1
        print('\nEarlyStopping: \n\tcurrent: %s\n\twait: %s \n\tbest: %s\n' % (current, self.wait, self.best))

def only_small_chi2(path='/home/fe918130/13TeV_chi2_disjoint_2',split=0.8):
    d=np.genfromtxt(path)[:,1:]
    p=d[:,:-1]
    chi2 = d[:,-1]
    mask = chi2 < 100
    p=p[mask]
    chi2 =chi2[mask]
    chi2 = chi2/100
    for i in range(p.shape[1]):
        mean,std=p[:,i].mean(), p[:,i].std()
        p[:,i] = (p[:,i] - mean)/std
    split = int(len(chi2)*split)
    x_train = p[:split]
    y_train = chi2[:split]
    x_test = p[split:]
    y_test = chi2[split:]
    return x_train, x_test, y_train, y_test

def patience_reduction(patience, lr_epoch):
    return max(5, int(patience/lr_epoch**(0.5)))

def get_error(error):
    if type(error) is list:
        return error[1] #index [0,1] is [train,test]
    else:
        return error


"""

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.66, patience=13, min_lr=10**(-6.0), verbose=1)

opt = Nadam(lr=10**(-2.8), beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0)
#opt = SGD(lr=10**(-2.55555), momentum=0.9, decay=0)

model.compile(loss = 'mae', optimizer = opt, metrics=[mean_error_chi2])
model.fit(x.train,y.train, validation_data =(x.test, y.test), epochs = n,batch_size=1000, callbacks = [reduce_lr, modcp, history])
y.evaluation(x, model)


x_train, x_test, y_train, y_test = only_small_chi2()
#sys.exit()
learnrate=10**(-2.8)
while learnrate > 10**(-7.1):
	opt = Nadam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0)
	model.compile(loss='mae', optimizer=opt, metrics=[mean_error_chi2])
	model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=400, batch_size=1000, verbose=1, callbacks=[history,early_stopping,modcp])
	model.load_weights('bestnet.hdf5')
	learnrate /= 4
	print 'learnrate:', learnrate


"""
