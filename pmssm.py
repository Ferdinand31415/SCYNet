#from data import master, hard_cut, tanh_cut, square_cut

import numpy as np
from numpy.random import RandomState
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
#from keras.layers.advanced_activations import SReLU
#srel = SReLU(t_left_init='zero', a_left_init='glorot_uniform', t_right_init='glorot_uniform', a_right_init='one')
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Nadam, Adagrad
#from keras.layers.normalization import BatchNormalization
from keras.callbacks import History, ModelCheckpoint, EarlyStopping


from copy import deepcopy
path ='13TeV_chi2_disjoint_2'
#d = tanh_cut(path, s=75, e=100)
#d= hard_cut(path, preproc ='dm')
#d = square_cut(path, cut=100, delta=25,split = 3.0/4, preproc = 'smds')

from Preprocessor import pmssm, chi2, shuffle_data
split = 9.0/10
x = pmssm(preproc = ['log_norm','min_max'], split = split)
y = chi2(['square_cut','div_max'], [100,25], split= split)
#shuffle_data(x, y)

model = Sequential()
n, act = 350, 'relu'
model.add(Dense(n, init='glorot_uniform',
#		init='zero',
		activation=act,
		input_dim=x.train.shape[1]))
for i in range(4):
    model.add(Dense(n-0*i, init='glorot_uniform',activation=act))#, W_regularizer=l2(0.001)))
    model.add(Dropout(0.08))
model.add(Dense(1, init='glorot_uniform',activation=act))


import keras.backend as K

def mean_loss_chi2(y_true, y_pred):
    return 100*K.mean(K.abs(y_pred-y_true))

def trainsgd(l=1e-1, mom=0.9, dec=1e-4, nes=False,n=3,b=500,loss='mse'):
	sgd = SGD(lr=l, momentum=mom, decay=dec, nesterov=nes)
	model.compile(loss = loss, optimizer = sgd)
	model.fit(x.train,y.train,nb_epoch = n,batch_size=b)
	d.eval(model)

def trainnadam(lr=1e-1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004,n=1,b=500,loss='mse'):
	nadam = Nadam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, schedule_decay=schedule_decay)
	model.compile(loss = loss, optimizer = nadam, metrics=[mean_loss_chi2])
	model.fit(x.train,y.train, validation_data =(x.test, y.test), epochs = n,batch_size=b)
	y.evaluation(x,model)

#Define Early Stopping(If val_loss doeasn't decrease in 50 epochs stop)
early_stopping = EarlyStopping(monitor='val_loss', patience=50)

#Save the model if new val_los minimum
modcp = ModelCheckpoint("best_net_r_chi2.hdf5", monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

#History allows to look at stuff like Loss and Val_Loss later
history = History()


#trainsgd(l=10**(-2.2),mom=0.2,n=10)
#trainnadam(lr=10**(-4),n=30,schedule_decay=1e-2)
