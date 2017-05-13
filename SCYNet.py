from copy import deepcopy
from Preprocessor import pmssm, chi2, log_norm, div_max, min_max, sub_mean_div_std
from keras.models import model_from_json, load_model
import misc
import numpy as np
import sys

class SCYNet:
    '''generate a fast chi2 prediction for pmssm points'''
    def __init__(self, masks=[[0,100]], output='output/SCYNet.out', model='SCYNET.h5', hp='hyperpoint.txt'):
        self.masks = masks #if one wants to exclude certain chi2 from getting written to output.
        #for example, masks = [[60,90],[91,92]] will output only chi2 in this ranges
        self.outputfile = output #name of file to write results
        self.model = load_model(model) #generate keras model
        self.hp = misc.load_result_file(hp)[0] #hyperparameter+information on the preprocessing. we need this to apply to the input pmssm the user gives.
        self.verbose = True

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

    def preprocess(self):
        '''preprocesses the p11mssm points in exactly the same way done for the net as it was trained'''
        if self.verbose:
            for k,v in zip(hp.keys(), hp.values())
                print k,v
        self.pp_data = deepcopy(self.data)
        for func in self.hp['pp_pmssm']:
            if self.verbose: print 'func:',func
            preproc_info = self.hp['pmssmtrafo'][func]
            if self.verbose: print 'preproc_info', preproc_info
            self.pp_data = eval(func)(self.pp_data, preproc_info)
       
    def backtransform_chi2(self):
        '''backtransforms chi2'''

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
        else:
            raise Exception('method set_data must get np array as argument and have Shape (N,11)')
        self.preprocess()
   
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
        for m in self.masks:
            mask = (chi_squared >= m[0]) == (chi_squared <= m[1])
            chi_squared = chi_squared[mask]
        
            if mode == 'chi2_only':
                with open(self.outputfile, 'a') as file:
                    for i in range(len(chi_squared)):
                        file.write(str(chi_squared[i]) + '\n')
            elif mode == 'all':
                 with open(self.outputfile, 'a') as file:
                    for i in range(len(chi_squared)):
                        line=' '.join(map(str, self.data[mask][i]))+' '+str(chi_squared[i])+'\n'
                        file.write(line)
            elif mode == 'pmssm_only':
                 with open(self.outputfile, 'a') as file:
                    for i in range(len(chi_squared)):
                        line=' '.join(map(str, self.data[mask][i]))+'\n'
                        file.write(line)
     
    def needfunctionthatgivesonlythosepmssmpointswhicharestillneeded:
        pass          


if __name__ == '__main__':
    mask = [int(sys.argv[1]), int(sys.argv[2])]
    raw_data = sys.argv[3]
    SN = SCYNet(mask)
    SN.read_data(raw_data)
    #SN.read_data('/net/home/lxtsfs1/tpe/feiteneuer/13TeV_chi2_disjoint_2') #for testing
    pred = SN.predict()
    SN.write_output(mode = 'pmssm_only')
