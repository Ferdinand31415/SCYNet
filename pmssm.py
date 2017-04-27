import numpy as np
from numpy.random import RandomState
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Nadam, Adagrad
#from keras.layers.normalization import BatchNormalization
from keras.callbacks import History, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

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

from copy import deepcopy

from Preprocessor import pmssm, chi2, shuffle_data, fulldata

data = fulldata()

split = 9.0/10
use_only = range(11)

data.shuffle()
x = pmssm(data.data[:,:-1], preproc = ['log_norm','min_max'], split = split)
y = chi2(data.data[:,-1], preproc = ['square_cut','div_max'], params = [100,25], split= split)

model = Sequential()
n, act = 300, 'relu'
model.add(Dense(n, kernel_initializer='glorot_uniform',
#		kernel_initializer='zero',
		activation=act,
		input_dim=x.train.shape[1]))
for i in range(3):
    model.add(Dense(n-0*i, kernel_initializer='glorot_uniform',activation=act))#, W_regularizer=l2(0.001)))
    model.add(Dropout(0.05))
model.add(Dense(1, kernel_initializer='glorot_uniform',activation=act))


import keras.backend as K
def mean_error_chi2(y_true, y_pred):
    return 100*K.mean(K.abs(y_pred-y_true))

def trainsgd(l=1e-1, mom=0.9, dec=1e-4, nes=False,n=3,b=500,loss='mse'):
	sgd = SGD(lr=l, momentum=mom, decay=dec, nesterov=nes)
	model.compile(loss = loss, optimizer = sgd)
	model.fit(x.train,y.train,nb_epoch = n,batch_size=b)
	d.eval(model)

def trainnadam(lr=1e-1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004,n=1,b=500,loss='mse'):
	nadam = Nadam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, schedule_decay=schedule_decay)
	model.compile(loss = loss, optimizer = nadam, metrics=[mean_error_chi2])
	model.fit(x.train,y.train, validation_data =(x.test, y.test), epochs = n,batch_size=b)
	y.evaluation(x,model)

#Define Early Stopping(If val_loss doeasn't decrease in 50 epochs stop)
early_stopping = EarlyStopping(monitor='val_loss', patience=18, mode='min', verbose=1)

#Save the model if new val_los minimum
modcp = ModelCheckpoint("bestnet.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#History allows to look at stuff like Loss and Val_Loss later
history = History()

#Main Lerning Loop. Start with standard Learning rate, then train still val_loss stops decreasing. Then decrease learning rate and train again.


#'''
learnrate=10**(-2.8)
while learnrate > 10**(-7.1):
	opt = Nadam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0)
	model.compile(loss='mae', optimizer=opt, metrics=[mean_error_chi2])
	model.fit(x.train, y.train, validation_data=(x.test,y.test), epochs=400, batch_size=1000, verbose=1, callbacks=[history,early_stopping,modcp])
	model.load_weights('bestnet.hdf5')
	learnrate /= 4
	print 'learnrate:', learnrate
#'''


'''
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.66, patience=13, min_lr=10**(-6.0), verbose=1)

opt = Nadam(lr=10**(-2.8), beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0)
#opt = SGD(lr=10**(-2.55555), momentum=0.9, decay=0)

model.compile(loss = 'mae', optimizer = opt, metrics=[mean_error_chi2])
model.fit(x.train,y.train, validation_data =(x.test, y.test), epochs = n,batch_size=1000, callbacks = [reduce_lr, modcp, history])
y.evaluation(x, model)
'''
import sys
'''
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

#trainnadam(lr=10**(-3.5),n=5,schedule_decay=1e-2)
#trainsgd(l=10**(-2.2),mom=0.2,n=10)
