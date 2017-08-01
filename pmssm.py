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

L = len(sys.argv)
mode = sys.argv[1] #'hp' or 'custom'


import keras.backend as K
import misc

#print 10*'_', '\n\n', 'cuda_vis_devices:', os.environ['CUDA_VISIBLE_DEVICES'], '\n\n'

from copy import deepcopy
from Preprocessor import pmssm, chi2, shuffle_data, fulldata
from hyperparameter import HyperPar
from netbuilder import build_Sequential_RH, build_Optimizer_RH  

#data = fulldata(path='/home/fe918130/13TeV_chi2_disjoint_2.npy',use=range(12))
#data = fulldata(path='/home/fe918130/data/pmssm_chi2_usebest20.npy',use=range(2,14))
#data = fulldata(path='/home/fe918130/data/mod_arrid_pmssm_chi2_55_14Jul.npy',use=range(2,14))
data = fulldata(path='/home/fe918130/data/pmssm_chi_24_07_57sr.npy',use=range(12))
#a = np.random.randint(1,10**15)
net = './output/pmssm_bestnet.h5'#.%s.hdf5' % a
print 'will save temporarily best net to %s' % net
if os.path.isfile(net):
    os.remove(net)

modcp = ModelCheckpoint(net, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = History()


if mode == 'custom':
    split = 9.0/10 #training/validiation split
    use_only = range(11) #use full pmssm set
    learnrate=10**(-2.8)
    initial_patience = 260

    data.shuffle()
    x = pmssm(data.data[:,:-1], preproc = ['sub_mean_div_std','div_max'], split = split)
    y = chi2(data.data[:,-1], preproc = ['square_cut','div_max'], params = [100,40], split= split)

    model = Sequential()
    n, act, init = 800, 'relu', 'normal'
    alpha=0.300007332516 #for Leaky Relu
    n = [600,20,300,300]
    model.add(Dense(n[0], kernel_initializer=init,
    #		kernel_initializer='zero',
            activation=act,
            input_dim=x.train.shape[1]))
    model.add(LeakyReLU(alpha=alpha))
    for i in n[1:]:
        model.add(Dense(i, kernel_initializer=init,activation=act))#, W_regularizer=l2(0.001)))
        model.add(Dropout(0.0744391624395))
        model.add(LeakyReLU(alpha=alpha))
        #model.add(PReLU())
    model.add(Dense(1, kernel_initializer=init,activation='relu')) #TODO try linear


    opt = Nadam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0)
    #opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mae', optimizer=opt)
elif mode =='hp':
    hp_txt, line = sys.argv[2], int(sys.argv[3])
    #hp_txt = path to txt file with my hpars.
    #line = actual line in the file (starting from 1 ...)
    hp_dict = misc.load_result_file(hp_txt)[line-1]

    '''
    #hpar = hp_dir +'/'+hp_dir.split('/')[-1]+'.txt'
    #print 'loading', hpar
    #hp_dict = misc.load_result_file(hpar)[0]
    '''
    hp = HyperPar(mode='set', hpar=hp_dict)
    hp.cut = 90
    print hp
    model = build_Sequential_RH(hp)
    opt = build_Optimizer_RH(hp)
    model.compile(loss='mae', optimizer=opt)
    initial_patience = 250
    learnrate = hp.lr
    #shuffle data, so we dont learn hyperparameters for a certain validation set
    data.shuffle(seed=hp_dict['randomseed'])
    split = 0.9
    x = pmssm(data.data[:,:-1], preproc = hp.pp_pmssm, split = split)
    y = chi2(data.data[:,-1], preproc = hp.pp_chi2, params = [hp.cut,hp.delta], split = split,verbose=False)
 

lr_epoch = 1
min_learnrate = 10**(-7.1)
lr_red = 3.5
while learnrate > min_learnrate:
    patience = max(2, int(initial_patience/(lr_epoch)**(0.4))) #patience decrease 
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1)
    K.set_value(model.optimizer.lr, learnrate)
    try:
        model.fit(x.train, y.train, validation_data=(x.test,y.test), epochs=1000, batch_size=1000, callbacks=[history,early_stopping,modcp],verbose=1)
    except KeyboardInterrupt:
        user_input = raw_input('What do you want to do? (q)uit or reduce lr (give factor)\n')
        if user_input == 'q':
            sys.exit()
        else:
            try:
                lr_red = float(user_input)     
            except:
                print 'could not convert to float, divide by standard value'
                lr_red = 3.5
            print 'manually reducing learnrate by %s hehehe...' % lr_red
            min_learnrate /= 2
    model.load_weights(net)
    learnrate /= lr_red
    lr_epoch += 1
    y.evaluation(x, model)

 
    print 'learnrate:%s, lr_epoch:%s, patience:%s' % (learnrate, lr_epoch, patience)

y.evaluation(x, model)
y.save()

#if sys.argv[1] == 'save':
if y.mean_errors['0.0-100.0'][1] < 1.3:
    misc.savemod(model, x, y)

os.remove(net)
