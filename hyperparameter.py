import itertools as it
import sys
import numpy as np
from numpy.random import RandomState
import time
from netbuilder import save_model
try:
    import cPickle as pickle
except:
    import pickle


class Hyperparameter():
    
    def __init__(self, paramfile='paramfile.pkl', verbose=True):
        self.paramfile = paramfile
        self.verbose = verbose
        try:
            self.load(self.verbose)
            self.create_todo() #creates list self.todo
        except IOError:
            print 'no file found!'
            #sys.exit()
            self.init()
            print 'created a new file %s' % self.paramfile
        
        self.create_todo() #creates self.todo, a list
 
    def __str__(self):
        msg  = '\nHyperparameter %s status \n' % self.paramfile
        wait, run, fin = 0, 0 ,0
        for i in self.p_keys():
            st = self.p[i]['status']
            if st == 'waiting':
                wait += 1
                continue
            if st == 'running':
                run += 1
            if st == 'finished':
                fin += 1
        msg += '\t waiting : %s\n' % wait
        msg += '\t running : %s\n' % run
        msg += '\t finished : %s\n' % fin
        msg += '\t total : %s\n' % len(self.p_keys())
        msg += '\t best mean error : %s ' % self.p['best_global_mean_err']
        
        return msg
  
     
    def create_layers(self):
        #create many good layers
        #TODO dont just manually enter all of them
        layers =[[50,50],[100,100,100],[300,300,300]]
        return layers

    def init(self):
        '''only do this once to initialize the paramaters.'''
        self.params = {}
        self.params['optimizer']=['nadam']
        #self.params['lr']=[0.1,0.01,0.001] #do this one interactively while training. not needed here
        self.params['layers']= self.create_layers()
        self.params['activation']=[['relu','relu','relu']]
        self.params['status'] = ['waiting']
        #self.params['regularization'] = ['l2','dropout']
        self.params['dropout'] = [0.01, 0.05, 0.1, 0.2, 0.4]
        self.params['meanerrors'] = ['tbd']
        self.params['init'] = ['glorot_uniform']
        self.params['batch'] = [500,1250]
        varNames = sorted(self.params)
        self.param = [dict(zip(varNames, prod)) for prod in it.product(*(self.params[varName] for varName in varNames))]
        from random import shuffle
        shuffle(self.param)
        self.len = len(self.param)
        self.p = {}
        for i in range(self.len):
            self.p.update({i:self.param[i]})
        self.p.update({'best_global_mean_err' : np.inf})
        self.save()
    
    def save(self):
        pickle.dump(self.p,open(self.paramfile,'wb'))
    def load(self, verbose=False):
        if verbose: print 'loading' , self.paramfile
        self.p = pickle.load(open(self.paramfile,'rb'))
    
    def p_keys(self):
        '''returns all ids, accessed via self.p[id]'''
        return [i for i in self.p.keys() if type(i) == int]

    def create_todo(self):
        def create_list(self):
            todo = []
            for i in self.p_keys():
                if self.p[i]['status'] == 'waiting':
                    todo.append(i)
            return todo 
        while True:
            todo_1 = create_list(self)
            #check times to see if other process wrote to file
            time.sleep(0.2)
            self.load(verbose=False)
            todo_2 = create_list(self)
            if todo_1 == todo_2:
                self.todo = todo_1
                break 
    
    '''change status of hp point
    if status is waiting we want to be able
    to test this hp point.

    if status is running/finished
    we do not want to touch it!
    '''
    def running(self, id):
        self.p[id]['status'] = 'running'
        self.save()
    def waiting(self, id):
        self.p[id]['status'] = 'waiting'
        if self.verbose: print '\nATTENTION: hp[%s] set to waiting' % id
        self.save()
    def finished(self, id, mean_errs, model=None):
        self.p[id]['status'] = 'finished'
        self.p[id]['meanerrors'] = mean_errs
        #mean_errs[range][0 or 1], train or test mean errors
        if mean_errs['0.0-100.0'][1] < self.p['best_global_mean_err']:
            print 'in finished'
            print self.p[id]['meanerrors'], mean_errs
            print mean_errs['0.0-100.0'], self.p['best_global_mean_err']
 
            self.p['best_global_mean_err'] = mean_errs['0.0-100.0']
            if model != None:
                save_model(model, name='best_ever')
            if self.verbose: print '\n\t!!!new best model!!! %s\n' % mean_errs
        self.save()
    
    def print_me(self):
        print '\n complete status'
        print self
        print 'details:'
        for i in self.p.keys():
            print self.p[i]

    def print_par(self, id):
        print '\nhyperparameter[%s]\n' % id
        for a, b in zip(self.p[id].keys(), self.p[id].values()):
            print '\t%s \t%s' %  (a, b)
        print '\n'

from random import uniform, randint
class RandomHyperPar()
    def __init__(self):
        self.lr= 10**(-uniform(1,6))
        self.dropout = uniform(0.01,0.4)
        self.batch = uniform(300,1500)
        self.layers = randint(3,7)
        self.neurons = randint(300,1000)
    
        #probability for choosing different types
        #of preprocessing for the pmssm
        prob = randint(1,2)
        if prob == 1:
            self.preproc_pmssm = 'log_norm'
        elif prob == 2:
            self.preproc_pmssm = 'sub_mean_div_std'

        #same for chi2 data
        prob = randint(1,2)
        if prob == 1:
            self.preproc_chi2 = 'div_max'
        elif prob == 2:
            self.preproc_chi2 = 'sub_mean_div_std'

        #same for optimizer
        prob = randint(1,2)
        if prob ==1:
            self.opt= 'sgd'
        elif prob == 2:
            self.opt= 'adam'
 
#test
#h = Hyperparameter()
#mean_errs = {'0.0-100':1.5, '0-50':2.0, '50-100':3.0}

h = RandomHyperPar()
