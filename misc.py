import numpy as np
def write_to_file(path_to_file,line):
    f = open(str(path_to_file), 'a')# a for append
    f.write(line)
    f.close()

def smart_start(x, y, model, batch):
    try_out = np.logspace(-1.5,-3.5,10)
    histos = []
    history = History()
    for lr in try_out:
        K.set_value(model.optimizer.lr, lr)
        model.fit(x.train, y.train, validation_data=(x.test,y.test),\
            epochs=3, batch_size=batch, verbose=1,\
            callbacks=[history,early_stopping,modcp])

def bad_loss(first_rounds):
    loss = first_rounds.history['loss'][0]
    if loss > 0.75 or np.isnan(loss) or np.isinf(loss):
        return True
    else:
        return False

def quit_early(histos):
    return almost_no_improvement(histos)


def almost_no_improvement(histos):
    '''returns true if almost no improvement at all even though
    we had smaller lr. We can then abort learning for time reasons
    in this case'''
    if len(histos) < 2:
        return False
    h_old = histos[-2].history['val_loss']
    h_new = histos[-1].history['val_loss']
    if min(h_new) / min(h_old) < 0.05:
        return True
    else:
        return False

import keras.backend as K
def mean_loss_chi2(y_true, y_pred):
    return 100*K.mean(K.abs(y_pred-y_true))


from keras.callbacks import EarlyStopping
class troll(EarlyStopping):         
    def __init__(self, patience=4):
        super(troll, self).__init__()
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



'''
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
'''
