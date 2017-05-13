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

mode = sys.argv[1] #'hp' or 'custom'

import keras.backend as K
import misc

#print 10*'_', '\n\n', 'cuda_vis_devices:', os.environ['CUDA_VISIBLE_DEVICES'], '\n\n'

from copy import deepcopy
from Preprocessor import pmssm, chi2, shuffle_data, fulldata
from hyperparameter import RandomHyperPar
from netbuilder import build_Sequential_RH, build_Optimizer_RH  

data = fulldata()

#a = np.random.randint(1,10**15)
net = './output/bestnet.h5'#.%s.hdf5' % a
print 'saving net temporarily to %s' % net
if os.path.isfile(net):
    os.remove(net)

modcp = ModelCheckpoint(net, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = History()


if mode == 'custom':
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


    opt = Nadam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0)
    #opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mae', optimizer=opt)
elif mode =='hp':
    hp_string = "layers?6;opt?nadam;beta_2?0.998788192552;alpha?0.204915674848;pp_chi2?['square_cut', 'div_max'];dropout?0.0243490131047;batch?1500;beta_1?0.691573565528;init?normal;neurons?[189, 240, 357, 400, 400, 361];lr?0.000447133593015;act?LReLU;pp_pmssm?['sub_mean_div_std', 'div_max'];initialpatience?65;chi2trafo?{'maxi': 100.0};pmssmtrafo?{'div_max:maxi': [2.5708434497879837, 2.9999020987586804, 3.1087680015109282, 2.935648079072779, 2.7869388695481749, 2.46911900946861, 2.6106053496039334, 2.4737363788055462, 2.5159150148568132, 2.593411738101262, 2.6952872526679776], 'sub_mean_div_std:meanstd': [[16.891927850694834, 1562.0038169581524], [1247.7163849872272, 917.38722271354902], [676.96779324500346, 1502.88543665666], [1979.7618628224118, 1025.9046305437362], [2242.1864078710028, 966.65400264260393], [1393.5286255485496, 650.55863350918469], [1759.817118776672, 858.09362193837148], [1694.8915134974677, 931.80653779912166], [89.763038121751251, 2022.998331758602], [-71.123783007872746, 1955.1751294533115], [23.485782177014869, 13.543999070878268]]};error?1.23026571644"
    hp_dict = misc.result_string_to_dict(hp_string)
    hp = RandomHyperPar()
    hp.set(hp_dict)
    print hp
    model = build_Sequential_RH(hp)
    opt = build_Optimizer_RH(hp)
    model.compile(loss='mae', optimizer=opt)
    initial_patience = 250
    learnrate = hp.lr
    #shuffle data, so we dont learn hyperparameters for a certain validation set
    data = fulldata()
    data.shuffle()
    split = 0.9
    x = pmssm(data.data[:,:-1], preproc = hp.pp_pmssm, split = split)
    y = chi2(data.data[:,-1], preproc = hp.pp_chi2, params = [100,25], split = split,verbose=False)

 

lr_epoch = 1
while learnrate > 10**(-7.1):
    patience = max(2, int(initial_patience/(lr_epoch)**(0.4))) #patience decrease 
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1)
    K.set_value(model.optimizer.lr, learnrate)
    model.fit(x.train, y.train, validation_data=(x.test,y.test), epochs=1000, batch_size=1000, callbacks=[history,early_stopping,modcp],verbose=1)
    model.load_weights(net)
    learnrate /= 3.5
    lr_epoch += 1
 
    print 'learnrate:%s, lr_epoch:%s, patience:%s' % (learnrate, lr_epoch, patience)

y.evaluation(x, model)

if sys.argv[1] == 'save':
    if y.mean_errors['0.0-100.0'][1] < 1.3:
        misc.savemod(model, x, y)
