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
class RandomHyperPar():
    def __init__(self, fasttrain=False):
        self.lr = 10**(-uniform(2.5,3.8))
        self.dropout = 10**(-uniform(2.0,1.0))#0.45
        self.batch = randint(500,2500)
        self.layers = randint(2,6)

        neurons_low, neurons_high = 150, 400
        if self.layers == 2:
            self.neurons = 2*[randint(neurons_low, neurons_high)]
        else:
            if randint(0,10) < 6:
               #block architecture
               self.neurons = self.layers*[randint(neurons_low, neurons_high)]
            else:
                #getting bigger and smaller!
                switch = randint(2,self.layers-1)
                self.neurons = self.layers*[0]
                self.neurons[0] = randint(neurons_low, int(neurons_high/2))
                for i in range(1,switch):
                    self.neurons[i] = int(min(neurons_high,uniform(1.1,1.5)*self.neurons[i-1]))
                for i in range(switch, self.layers):
                    self.neurons[i] = int(max(neurons_low,self.neurons[i-1]/uniform(1.1,1.5)))
        if fasttrain:
            self.layers = randint(2,3)
            self.neurons = self.layers*[randint(20,80)]
            #____
        
        #probability for choosing different types
        #of preprocessing for the pmssm

        p = randint(0,6)
        if p == 0:
            self.pp_pmssm = ['log_norm','sub_mean_div_std']
        elif p == 1:
            self.pp_pmssm = ['log_norm','min_max']
        elif p == 2:
            self.pp_pmssm = ['log_norm','div_max']
        elif p == 3:
            self.pp_pmssm = ['min_max']
        elif p == 4:
            self.pp_pmssm = ['sub_mean_div_std','div_max']
        elif p == 5:
            self.pp_pmssm = ['sub_mean_div_std','min_max']
        elif p == 6:
            self.pp_pmssm = ['div_max']
            
        p = randint(0,2)
        #same for chi2 data
        if p == 0:
            self.pp_chi2 = ['square_cut','div_max']
        elif p == 1:
            self.pp_chi2 = ['square_cut','sub_mean_div_std']
        elif p == 2:
            self.pp_chi2 = ['square_cut','min_max']

        prob = [randint(0,1) for p in range(10)]

        #same for optimizer
        if randint(0,10) < 0: #have not rly used sgd, so try only few
            self.opt= 'sgd'
            self.lr *= 7 #often, the lr is much too small
        else:    
            self.opt= 'nadam'
            #beta_1=0.9, beta_2=0.999 standard values
            self.beta_1 = uniform(0.5, 0.99)
            self.beta_2 = uniform(0.99, 0.9999)


        #kernel initializer
        if randint(0,10) < 6:
            self.init = 'glorot_uniform'
        else:
            self.init = 'normal'

        #activation function
        #tanh never works well... atm only RELU's
        #try ELU? exponential linear unit
        if randint(0,10) < 10:
            self.act = 'LReLU'
            self.alpha = 10**(-uniform(2, 0.46))
        else: #this one did not seem so good?
            self.act = 'PReLU'

    def __str__(self):
        s = '\nhyperparameter:'
        hp = self.__dict__
        for value, key in zip(hp.values(), hp.keys()):
            s += '\n\t' + str(key) + '\t' + str(value)
        return s

    def string(self):
        s = ''
        hp = self.__dict__
        for value, key in zip(hp.values(), hp.keys()):
            s += str(key) + '?' + str(value) + ';'
        return s

    def dict(self):
        return self.__dict__

    def set_(self, lr, dropout, batch, layers, neurons, opt, pp_pmssm, pp_chi2):
        self.lr = lr
        self.dropout = dropout
        self.batch = batch
        self.layers = layers
        self.neurons = neurons
        self.opt = opt
        self.pp_pmssm = pp_pmssm
        self.pp_chi2 = pp_chi2

    def set(self, hp_dict):
        for key, value in zip(hp_dict.keys(), hp_dict.values()):
            setattr(self, key, value)
 
    def vary(self):
        self.batch += randint(-40,40)
        self.neurons += randint(-20,20)
        self.dropout *= uniform(0.8,1.2)
        #TODO ADAM 

#test
#h = Hyperparameter()
#mean_errs = {'0.0-100':1.5, '0-50':2.0, '50-100':3.0}

#h = RandomHyperPar()
