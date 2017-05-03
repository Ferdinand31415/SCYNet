from copy import deepcopy
from Preprocessor import pmssm, chi2
from keras.models import model_from_json
import numpy as np

class SCYNet:
    '''generate a fast chi2 prediction for pmssm points'''
    def __init__(self, mask=None, output=None):
        self.mask = mask #if one wants to exclude certain chi2 from getting written to output.
        #for example, mask = [60,90] will output only chi2 in this range
        self.output = output #name of file to give results
        self.load_model() #generate keras model


    def load_model(self, name='SCYNet'):
        '''generates a keras neural net model'''
        # load json and create model
        json_file = open('%s.json' % name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("%s.h5" % name)
        print("Loaded model %s from disk" % name)
        self.model = loaded_model
        return loaded_model

    def preprocess(self):
        '''preprocesses pmssm 11 data like it has been done
        in the training phase'''
        self.maxis = np.load('preprocess_info.npy')
        self.pp_data = deepcopy(self.data)
        for i in range(11):
            self.pp_data[:,i] /= self.maxis[i]

    def predict(self, data=None):
        if data == None:
            if hasattr(self, 'data'):
                self.pred = self.model.predict(self.pp_data)
        elif isinstance(data, np.ndarray) and data.shape[1] == 11:
            self.data = data
            self.preprocess() #preprocesses the just arrived self.data
            self.pred = self.model.predict(self.pp_data)
        else:
            raise Exception('SCYNet has no pmssm data with shape(N,11) to predict chi2')
        return self.backtransform_chi2()

    def backtransform_chi2(self):
        return chi2

    def set_data(self, data):
        '''data must be of type np array'''
        if isinstance(data, np.ndarray) and data.shape[1] == 11:
            self.data = data
            self.preprocess()
        else:
            raise Exception('method set_data must get np array as argument and have Shape (N,11)')
    
    def read_data(self, inputfile):
        '''reading in the data. test if numpy file ('something.npy') or txtfile'''
        try:
            self.data = np.load(inputfile)
        except(IOError):
            pass
        
        try:
            self.data = np.genfromtxt(inputfile)
        except:
            print('You must provide either a numpy array'
                  'or whitespace seperated textfile')
        if self.data.shape[1] != 11:
            raise ValueError('data has wrong shape, shape=%s' % self.data.shape)        

    def write_output(self, mode='all'):
        '''writes output to file'''
        if self.output == None:
           self.output = 'SCYNet.output'

        if self.mask != None:
            chi2 = deepcopy(self.pred)
            mask = (chi2 > self.mask[0]) == (chi2 < self.mask[1])
            chi2 = chi2[mask]
        else:
            chi2 = self.pred
    
        if mode == 'chi2only':
            with open(self.output, 'w') as file:#destroys any existing result_file
                for i in range(len(chi2)):
                    file.write(str(chi2[i]) + '\n')
        else:
             with open(self.output, 'w') as file:#destroys any existing result_file
                for i in range(len(chi2)):
                    line=' '.join(map(str, self.data[i]))+' '+str(chi2[i])+'\n'
                    file.write(line)
            


