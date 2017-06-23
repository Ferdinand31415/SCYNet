from copy import deepcopy
from Preprocessor import pmssm, chi2, log_norm, div_max, min_max, sub_mean_div_std
from keras.models import model_from_json, load_model
import misc
import numpy as np
import sys
import time

class SCYNet:
    '''generate a fast chi2 prediction for pmssm points'''
    def __init__(self, args, model='SCYNET.h5', hp='hyperpoint.txt'):
        self.model = load_model(model) #generate keras model
        self.hp = misc.load_result_file(hp)[0] #hyperparameter+information on the preprocessing. we need this to apply to the input pmssm the user gives.
        for i in range(len(args)):
            if args[i] =='-data' or args[i] == '-d':
                self.datapath = args[i+1]
            if args[i] == '-m' or args[i] == '-mask':
                #for example, masks = [[60,90],[91,92]] will output only chi2 in this ranges
                self.masks = [map(float, arg.split('-')) for arg in args[i+1].split(',')]
            if args[i] == '-o' or args[i] == '-output':
                self.outputfile = str(args[i+1])
            if args[i] == '-v' or args[i] == '-verbose':
                self.verbose=True
            if args[i] == '-i' or args[i] == '-include':
                try:
                    self.include_additionally = np.load(args[i+1])
                except:
                    print 'need valid numpy file, could not load your desired data'
            if args[i] == '-w' or args[i] == '-write':
                self.mode = args[i+1]
        if not hasattr(self, 'outputfile'):
            self.outputfile = 'output/SCYNet.out'
        if not hasattr(self, 'masks'):
            self.masks = [[0,100]]
        if not hasattr(self, 'verbose'):
            self.verbose = False
        if not hasattr(self, 'mode'):
            self.mode = 'all'
        if hasattr(self, 'datapath'):
            self.read_data(self.datapath)
        print 'initialized SCYNet, your options:'
        print '\tmasks:', self.masks
        print '\toutputfile:', self.outputfile
        if hasattr(self, 'datapath'):
            print '\tinput data:', self.datapath
        print '\tverbose:', self.verbose

    def predict(self, data=None):
        if data == None:
            if hasattr(self, 'pp_data'):
                self.pred = self.model.predict(self.pp_data)
            else:
                raise Exception('SCYNet has no attribute "pp_data" expected at this point')
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
                self.data = np.genfromtxt(inputfile)#[:,1:-1] #for testing 13TeV_chi2_disjoint_2
            except:
                print('You must provide either a numpy array'
                      'or whitespace seperated textfile')
        if self.data.shape[1] != 11:
            raise ValueError('data has wrong shape, shape=%s' % self.data.shape)
        if self.verbose: print '\tinput data:', inputfile
        self.preprocess()

    def write_output_old(self, mode='all'):
        '''writes output to file'''
        written = 0
        if self.verbose: print 'writing output ... mode: %s' % self.mode
        with open(self.outputfile, 'w') as file:
            for m in self.masks:
                chi_squared = deepcopy(self.pred)
                mask = (chi_squared >= m[0]) == (chi_squared <= m[1])
                chi_squared = chi_squared[mask]
                
                if self.mode == 'chi2_only':
                    for i in range(len(chi_squared)):
                        file.write(str(chi_squared[i]) + '\n')
                elif self.mode == 'all':
                    for i in range(len(chi_squared)):
                        line=' '.join(map(str, self.data[mask][i]))+' '+str(chi_squared[i])+'\n'
                        file.write(line)
                elif self.mode == 'pmssm_only':
                    for i in range(len(chi_squared)):
                        line=' '.join(map(str, self.data[mask][i]))+'\n'
                        file.write(line)
                else:
                    raise IOError('wrong mode, can only have "chi2_only","pmssm_only", or "all"')
                if self.verbose: print 'points in %s: %s' % (m, len(chi_squared))
                written +=len(chi_squared)

        if self.verbose: print 'wrote %s lines to %s' % (written, self.outputfile)

    def write_output(self,mode='all'):
        '''writes output to file
        '''
        if self.verbose: print 'writing output ... mode: %s' % self.mode
        output = ''
        written = 0
        self.masks_ = []
        self.total_mask = np.zeros(len(self.pred))
        for m in self.masks:
            chi_squared = deepcopy(self.pred)
            mask = (chi_squared >= m[0]) == (chi_squared < m[1])
            chi_squared = chi_squared[mask]
            self.total_mask += mask.astype(int)
            self.masks_.append([mask,chi_squared])
            print 'sum', sum(mask.astype(int))
            if self.mode == 'chi2_only':
                for i in range(len(chi_squared)):
                    output += str(chi_squared[i]) + '\n'
            elif self.mode == 'all':
                for i in range(len(chi_squared)):
                    line=' '.join(map(str, self.data[mask][i]))+' '+str(chi_squared[i])+'\n'
                    output += line
            elif self.mode == 'pmssm_only':
                for i in range(len(chi_squared)):
                    line=' '.join(map(str, self.data[mask][i]))+'\n'
                    output += line
            else:
                raise IOError('wrong mode, can only have "chi2_only","pmssm_only", or "all"')
            written += len(chi_squared)
            if self.verbose: print 'will write %s lines in mask %s' % (len(chi_squared), m)
        
        with open(self.outputfile, 'w') as file:
            file.write(output)
            if self.verbose: print 'wrote %s lines' % written
            if hasattr(self, 'include_additionally'):
                if self.verbose: print 'writing additional output according to given numpy array'
                #example for weird formula below
                #total mask= [1,1,1,0,0]
                #incl addit= [0,0,1,1,0]
                #-> add mask=[0,0,0,1,0]
                self.include_additionally=self.include_additionally.astype(int)
                additional_mask = (self.include_additionally - self.total_mask + 1 ) / 2
                additional_mask = additional_mask.astype(bool)
                output=''
                chi_squared = self.pred[additional_mask]
                data = self.data[additional_mask]
                if self.mode == 'chi2_only':
                    for i in range(len(chi_squared)):
                        output += str(chi_squared[i]) + '\n'
                elif self.mode == 'all':
                    for i in range(len(chi_squared)):
                        line=' '.join(map(str, data[i]))+' '+str(chi_squared[i])+'\n'
                        output += line
                elif self.mode == 'pmssm_only':
                    for i in range(len(chi_squared)):
                        line=' '.join(map(str, self.data[i]))+'\n'
                        output += line
                else:
                    raise IOError('wrong mode, can only have "chi2_only","pmssm_only", or "all"')
                file.write(output)
                if self.verbose: print 'wrote %s more lines from additional input' % len(chi_squared)
    def needfunctionthatgivesonlythosepmssmpointswhicharestillneeded(self):
        pass          

if __name__ == '__main__':
    #chosing SCYNet
    path = '/home/fe918130/resultSCYNet/nets/'
    net = '1.3728_15May'
    net = path + net + '/' + net

    SN = SCYNet(args=sys.argv, model=net+'.h5', hp=net+'.txt')
    pred = SN.predict()

    include = np.load('/home/fe918130/data/acceptMRT_23_06.npy')
    SN.write_output(mode = 'all')
