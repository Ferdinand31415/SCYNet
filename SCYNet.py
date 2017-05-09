from copy import deepcopy
from Preprocessor import pmssm, chi2
from keras.models import model_from_json, load_model
import numpy as np
import sys

class SCYNet:
    '''generate a fast chi2 prediction for pmssm points'''
    def __init__(self, mask=[0,100], output='output/SCYNet.out'):
        self.mask = mask #if one wants to exclude certain chi2 from getting written to output.
        #for example, mask = [60,90] will output only chi2 in this range
        self.output = output #name of file to give results
        self.model = load_model('SCYNet.h5') #generate keras model
        #self.y = chi2(data=None, preproc=[100,25], params=None, split=1)

    def load_json_model(self, name='SCYNet'):
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
            self.preprocess() #preprocesses the just arrived self.data to self.pp_data
            self.pred = self.model.predict(self.pp_data)
        else:
            raise Exception('SCYNet has no pmssm data with shape(N,11) to predict chi2')
        self.pred = self.pred.flatten()
        return self.backtransform_chi2() #need to backtransform chi2 into 0..100 range

    def backtransform_chi2(self):
        #1. mult_max
        self.pred *= 100
        #2. back_square
        cut, delta = [100, 25]
        y = self.pred
        mask = (y > cut-delta) == (y < cut) #backtrafo needs a different mask than trafo
        y[mask]= (cut+delta)-2*np.sqrt(delta*(cut-y[mask]))
        y[y>cut] = cut
        self.pred = y
        return self.pred

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
            try:
                self.data = np.genfromtxt(inputfile)#[:,1:-1] #for testing 13TeV_chi2_disjoint_2
            except:
                print('You must provide either a numpy array'
                      'or whitespace seperated textfile')
        if self.data.shape[1] != 11:
            raise ValueError('data has wrong shape, shape=%s' % self.data.shape)
        self.preprocess()
        
    def write_output(self, mode='all'):
        '''writes output to file'''
        chi_squared = deepcopy(self.pred)
        mask = (chi_squared >= self.mask[0]) == (chi_squared <= self.mask[1])
        chi_squared = chi_squared[mask]
    
        if mode == 'chi2_only':
            with open(self.output, 'w') as file:#destroys any existing result_file
                for i in range(len(chi_squared)):
                    file.write(str(chi_squared[i]) + '\n')
        elif mode == 'all':
             with open(self.output, 'w') as file:#destroys any existing result_file
                for i in range(len(chi_squared)):
                    line=' '.join(map(str, self.data[mask][i]))+' '+str(chi_squared[i])+'\n'
                    file.write(line)
        elif mode == 'pmssm_only':
             with open(self.output, 'w') as file:#destroys any existing result_file
                for i in range(len(chi_squared)):
                    line=' '.join(map(str, self.data[mask][i]))+'\n'
                    file.write(line)
 
            


if __name__ == '__main__':
    mask = [int(sys.argv[1]), int(sys.argv[2])]
    raw_data = sys.argv[3]
    SN = SCYNet(mask)
    SN.read_data(raw_data)
    #SN.read_data('/net/home/lxtsfs1/tpe/feiteneuer/13TeV_chi2_disjoint_2') #for testing
    pred = SN.predict()
    SN.write_output(mode = 'pmssm_only')
