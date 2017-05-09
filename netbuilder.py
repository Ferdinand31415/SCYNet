from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, Nadam, Adagrad
from keras.layers.advanced_activations import LeakyReLU, PReLU

#for saving and loading models
from keras.models import model_from_json

def build_Sequential(hp):
    model = Sequential()
    
    #input layer
    '''
    model.add(Dense(hp['layers'][0], init=hp['init'],\
                    activation = hp['activation'][0],
                    input_dim = 11))
    '''
    model.add(Dense(hp['layers'][0], input_dim=11,\
                    kernel_initializer=hp['init']))
    model.add(Activation(hp['activation'][0]))

   
    #hidden layers
    '''
    for i in range(1, len(hp['layers'])):
        model.add(Dense(hp['layers'][i],\
                        init=hp['init'],\
                        activation=hp['activation'][1]))
        model.add(Dropout(hp['dropout']))
    '''
    
    for i in range(1, len(hp['layers'])): #skipping first!
        model.add(Dense(hp['layers'][i],\
                        kernel_initializer=hp['init']))
        model.add(Dropout(hp['dropout']))
        model.add(Activation(hp['activation'][1]))


    #output layer
    #model.add(Dense(1, init=hp['init'], activation=hp['activation'][2]))
    model.add(Dense(1, kernel_initializer=hp['init']))
    model.add(Activation(hp['activation'][2]))
    return model

def build_Sequential_RH(hp):
    model = Sequential()
    model.add(Dense(hp.neurons, input_dim=11,\
                    kernel_initializer=hp.init))
    model.add(Activation(hp.act))

   
    #hidden layers
    for i in range(1, hp.layers): #skipping first!
        model.add(Dense(hp.neurons,\
                        kernel_initializer=hp.init))
        model.add(Dropout(hp.dropout))
        model.add(Activation('linear'))
        if hp.act == 'LReLU':
            model.add(LeakyReLU(alpha=hp.alpha))
        elif hp.act == 'PReLU':
            model.add(PReLU())# same as LReLU, but learns alpha

    #output layer
    model.add(Dense(1, kernel_initializer=hp.init))
    model.add(Activation('relu'))
    return model


def build_Optimizer(hp, starting_lr):
    if hp['optimizer'] == 'nadam':
        return Nadam(lr=starting_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0)
    else:
        raise ValueError('no optimizer was given inside %s' % hp)

def build_Optimizer_RH(hp):
    if hp.opt == 'nadam':
        return Nadam(lr=hp.lr, beta_1=0.9, beta_2=0.999, \
                    epsilon=1e-08, schedule_decay=0)
    if hp.opt == 'sgd':
        return SGD(lr=hp.lr, momentum=0.9, decay=0, nesterov=True) 

def save_model(model, name='best_ever'):
    # serialize model to JSON
    model_json = model.to_json()
    with open("%s.json" % name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("%s.h5" % name)
    print("Saved model %s to disk" % name)

def load_model(name='best_ever'):
    # load json and create model
    json_file = open('%s.json' % name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s.h5" % name)
    print("Loaded model %s from disk" % name)
    return loaded_model

