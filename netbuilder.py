from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, Nadam, Adagrad

#for saving and loading models
from keras.models import model_from_json

def build_Sequential(hp):
    model = Sequential()
    
    #input layer
    model.add(Dense(hp['layers'][0], init=hp['init'],\
                    activation = hp['activation'][0],
                    input_dim = 11))

    #hidden layers
    for i in range(1, len(hp['layers'])):
        model.add(Dense(hp['layers'][i],\
                        init=hp['init'],\
                        activation=hp['activation'][1]))
        model.add(Dropout(hp['dropout']))

    #output layer
    model.add(Dense(1, init=hp['init'], activation=hp['activation'][2]))
    return model

def build_Optimizer(hp):
    if hp['optimizer'] == 'nadam':
        return Nadam(lr=10**(-3), beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0)
    else:
        raise ValueError('no optimizer was given %s' % hp)

def save_model(model, name='best_ever')
    # serialize model to JSON
    model_json = model.to_json()
    with open("%s.json" % name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("%s.h5" % name)
    print("Saved model %s to disk" % name)

def load_model(name='best_ever')
    # load json and create model
    json_file = open('%s.json' % name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s.h5" % name)
    print("Loaded model %s from disk" % name)
    return loaded_model

