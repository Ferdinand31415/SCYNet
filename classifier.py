from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Nadam, Adagrad
#from keras.layers.normalization import BatchNormalization
from keras.callbacks import History, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import sys
import keras.backend as K

data = '/scratch/work/eiteneuer/data/numpy_data_pmssm'
data = '/home/fe918130'

x = np.load(data +'/pmssm.npy')
y = np.load(data +'/chi2_binary.npy')
y = y/100.0

split = int(len(x)*6.0/8)
x_test = x[split:]
x_train = x[:split]

y_test = y[split:]
y_train = y[:split]

from keras.utils import np_utils
y_train_ohe = np_utils.to_categorical(y_train)
y_test_ohe = np_utils.to_categorical(y_test)

import sys
#sys.exit()
mod = []
for i in range(x.shape[1]):
    std = x[:,i].std()
    mean = x[:,i].mean()
    x[:,i] = (x[:,i]-mean)/std
    mod.append([mean,std])


model = Sequential()
n, act = 1500, 'relu'
model.add(Dense(n, kernel_initializer='glorot_uniform',
#		kernel_initializer='zero',
		activation=act,
		input_dim=11))
for i in range(3):
    model.add(Dense(n-0*i, kernel_initializer='glorot_uniform',activation=act, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.1))

model.add(Dense(1, kernel_initializer='glorot_uniform',activation='sigmoid'))


#opt = SGD(lr=10**(-2.55555), momentum=0.9, decay=0, nesterov=True)
opt = Nadam(lr=10**(-2.5), beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0)
early_stopping = EarlyStopping(monitor='val_loss', patience=13, mode='min', verbose=1)
modcp = ModelCheckpoint("bestclassifier.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = History()
histos = []
lr=10**(-3.5)
if len(sys.argv) > 1:
    lr = 10**(float(sys.argv[-1]))

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
while lr > 10**(-7.1):
    K.set_value(model.optimizer.lr, lr)
    history = History()
    model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=400, batch_size=300, verbose=1, callbacks=[history,early_stopping,modcp])
    model.load_weights('bestclassifier.hdf5')
    lr /= 4.0
    print 'lr:', lr
    histos.append(history)

#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.66, patience=13, min_lr=10**(-6.0), verbose=1)


#model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=['accuracy'])
#model.fit(x_train,y_train, validation_data =(x_test, y_test), epochs = n,batch_size=1000, callbacks = [reduce_lr, modcp, history])
#'''
