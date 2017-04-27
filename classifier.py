from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Nadam, Adagrad
#from keras.layers.normalization import BatchNormalization
from keras.callbacks import History, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau




x = np.load('/scratch/work/eiteneuer/data/numpy_data_pmssm/pmssm.npy')
y = np.load('/scratch/work/eiteneuer/data/numpy_data_pmssm/chi2_binary.npy')

split = int(len(x)*6.0/8)
x_test = x[:split]
x_train = x[split:]

y_test = x[:split]
y_train = x[split:]

mod = []
for i in x.shape[1]:
    std = x[:,i].std()
    mean = x[:,i].mean()
    x[:,i] = (x[:,i]-mean)/std
    mod.append(mean,std)


model = Sequential()
n, act = 100, 'relu'
model.add(Dense(n, init='glorot_uniform',
#		init='zero',
		activation=act,
		input_dim=x.train.shape[1]))
for i in range(3):
    model.add(Dense(n-0*i, init='glorot_uniform',activation=act))#, W_regularizer=l2(0.001)))
    model.add(Dropout(0.08))

model.add(Dense(1, init='tanh',activation=act))


def trainnadam(lr=1e-1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004,n=1,b=500,loss='binary_crossentropy'):
	nadam = Nadam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, schedule_decay=schedule_decay)
	model.compile(loss = loss, optimizer = nadam, metrics=['accuracy'])
	model.fit(x_train,y_train, validation_data =(x_test, y_test), epochs = n,batch_size=b)

trainnadam(lr=10**(-3.5),n=5,schedule_decay=1e-2)



