from copy import deepcopy
from Preprocessor import pmssm, chi2, log_norm, div_max, min_max, sub_mean_div_std
from keras.models import model_from_json, load_model
import misc
import numpy as np
import sys

class SCYNet:
    '''generate a fast chi2 prediction for pmssm points'''
    def __init__(self, masks=[[0,100]], output='output/SCYNet.out', model='SCYNET.h5', hp='hyperpoint.txt', verbose=True):
        self.masks = masks #if one wants to exclude certain chi2 from getting written to output.
        #for example, masks = [[60,90],[91,92]] will output only chi2 in this ranges
        self.outputfile = output #name of file to write results
        self.model = load_model(model) #generate keras model
        self.hp = misc.load_result_file(hp)[0] #hyperparameter+information on the preprocessing. we need this to apply to the input pmssm the user gives.
        self.verbose = verbose

    def predict(self, data=None):
        if data == None:
            if hasattr(self, 'pp_data'):
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
            for k,v in zip(self.hp.keys(), self.hp.values()):
                print k,v
        self.pp_data = deepcopy(self.data)
        for func in self.hp['pp_pmssm']:
            if self.verbose: print '\n*****func:',func
            preproc_info = self.hp['pmssm_back_info'][func]
            if self.verbose: print '\n*****preproc_info', preproc_info
            self.pp_data = eval(func)(self.pp_data, preproc_info) #these functions got imported from Preprocessor above
       
    def backtransform_chi2(self):
        '''do the same transformations that were applied to the chi2 during the training backwards'''
        params=[self.hp['chi2_back_info']['cut'], self.hp['chi2_back_info']['delta']]

        #chi2 class has inbuilt functionality for the backtrafo.
        #It matters only that we init chi2 class with params,rest irrelevant
        y = chi2(data=np.array([]), preproc=[],params=params,split=0,verbose=True)
        y.back = self.hp['chi2_back_trafos']
        if self.verbose: print 'y.back:',y.back

        #the chi2 class needs accesses the attributes in this way.
        for attribute in self.hp['chi2_back_info']:
            value = self.hp['chi2_back_info'][attribute]
            if self.verbose: print 'setting chi2.%s to %s'%(attribute,value)
            setattr(y, attribute, value)

        #actual backtransformation
        self.pred = y.backtrafo(self.pred) #applys functions in backward order
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
                self.data = np.genfromtxt(inputfile)[:,1:-1] #for testing 13TeV_chi2_disjoint_2
            except:
                print('You must provide either a numpy array'
                      'or whitespace seperated textfile')
        if self.data.shape[1] != 11:
            raise ValueError('data has wrong shape, shape=%s' % self.data.shape)
        self.preprocess()

    def write_output(self, mode='all'):
        '''writes output to file'''
        chi_squared = deepcopy(self.pred)
        with open(self.outputfile, 'w') as file:
            for m in self.masks:
                mask = (chi_squared >= m[0]) == (chi_squared <= m[1])
                chi_squared = chi_squared[mask]
            
                if mode == 'chi2_only':
                    for i in range(len(chi_squared)):
                        file.write(str(chi_squared[i]) + '\n')
                elif mode == 'all':
                    for i in range(len(chi_squared)):
                        line=' '.join(map(str, self.data[mask][i]))+' '+str(chi_squared[i])+'\n'
                        file.write(line)
                elif mode == 'pmssm_only':
                    for i in range(len(chi_squared)):
                        line=' '.join(map(str, self.data[mask][i]))+'\n'
                        file.write(line)
                else:
                    raise ValueError('wrong mode, can only have "chi2_only","pmssm_only", or "all"')
     
    def needfunctionthatgivesonlythosepmssmpointswhicharestillneeded(self):
        pass          


if __name__ == '__main__':
    path = '/home/fe918130/resultSCYNet/nets/'
    net = '1.3728_15May'
    #test model, 1.6 err
    #path= '/home/fe918130/resultSCYNet/nets/1.6357_15May/model_1.6357_15May'
    net = path + net + '/' + net

    mask = [[int(sys.argv[1]), int(sys.argv[2])]]
    #raw_data = sys.argv[3]
    SN = SCYNet(masks=mask, model=net+'.h5', hp=net+'.txt')
    #SN.read_data(raw_data)
    #SN.read_data('/net/home/lxtsfs1/tpe/feiteneuer/13TeV_chi2_disjoint_2_test') #for testing
    SN.read_data('/home/fe918130/13TeV_chi2_disjoint_2')
    pred = SN.predict()
    SN.write_output(mode = 'chi2_only')
