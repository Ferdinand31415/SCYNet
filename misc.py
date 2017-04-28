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

