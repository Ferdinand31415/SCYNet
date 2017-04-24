from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, Nadam, Adagrad

def build_sequential(hp):
    model = Sequential()
    
    #input layer
    model.add(Dense(hp['layers'][0], init=hp['init'],\
                    activation = hp['activation'][0],
                    input_dim = 11))

    #hidden layers
    for i in range(1, len(hp['layers'])):
        model.add(Dense(hp['layers'][i],\
                        init=hp['init'],\
                        activation=hp['activation']))
        model.add(Dropout(hp['dropout'])

    #output layer
    model.add(Dense(1, init=hp['init'], activation=hp['activation'][1]))
    return model



