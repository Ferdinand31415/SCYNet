import numpy as np
from numpy.random import RandomState
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Nadam, Adagrad, RMSprop
#from keras.layers.normalization import BatchNormalization
from keras.callbacks import History, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import sys, os

import keras.backend as K
import misc

#print 10*'_', '\n\n', 'cuda_vis_devices:', os.environ['CUDA_VISIBLE_DEVICES'], '\n\n'

from copy import deepcopy
from Preprocessor import pmssm, chi2, shuffle_data, fulldata

data = fulldata()

split = 9.0/10 #training/validiation split
use_only = range(11) #use full pmssm set
learnrate=10**(-3.8)
initial_patience = 60

data.shuffle()
x = pmssm(data.data[:,:-1], preproc = ['div_max'], split = split)
y = chi2(data.data[:,-1], preproc = ['square_cut','div_max'], params = [100,25], split= split)

model = Sequential()
n, act, init = 3500, 'linear', 'glorot_uniform'
alpha=0.300007332516 #for Leaky Relu
model.add(Dense(n, kernel_initializer=init,
#		kernel_initializer='zero',
		activation=act,
		input_dim=x.train.shape[1]))
model.add(LeakyReLU(alpha=alpha))
for i in range(3):
    model.add(Dense(n-0*i, kernel_initializer=init,activation=act))#, W_regularizer=l2(0.001)))
    model.add(Dropout(0.0744391624395))
    model.add(LeakyReLU(alpha=alpha))
    #model.add(PReLU())
model.add(Dense(1, kernel_initializer=init,activation='relu')) #TODO try linear


#a = np.random.randint(1,10**15)
net = './output/bestnet.h5'#.%s.hdf5' % a
print 'saving net temporarily to %s' % net
if os.path.isfile(net):
    os.remove(net)

modcp = ModelCheckpoint(net, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = History()

opt = Nadam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0)
#opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='mae', optimizer=opt)

lr_epoch = 0
while learnrate > 10**(-7.1):
    early_stopping = EarlyStopping(monitor='val_loss', patience=80, mode='min', verbose=1)
    K.set_value(model.optimizer.lr, learnrate)
    model.fit(x.train, y.train, validation_data=(x.test,y.test), epochs=1000, batch_size=1000, callbacks=[history,early_stopping,modcp],verbose=1)
    model.load_weights(net)
    learnrate /= 3.5
    lr_epoch += 1
 
    patience = max(2, int(initial_patience/(lr_epoch)**(0.35))) #patience decrease 

    print 'learnrate:', learnrate, '\tAmount of times lr has beed adjusted:',lr_epoch

y.evaluation(x, model)

#if len(sys.argv) > 1 and sys.argv[1] == 'save':
if y.mean_errors['0.0-100.0'][1] < 1.3:
    misc.savemod(model, x, y)
