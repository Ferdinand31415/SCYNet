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
        self.scannedfile = 'alrdyscanned'#why did i create this?
        self.verbose = verbose
        try:
            self.load()
            self.create_todo() #creates list self.todo
        except:
            print 'no file found'
            #sys.exit()
            self.init()
            print 'created a new file %s' % self.paramfile
        
        self.create_todo() #creates self.todo, a list
 
    def __str__(self):
        msg  = '\nHyperparameter %s status \n' % self.paramfile
        wait, run, fin = 0, 0 ,0
        for i in self.p.keys():
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
        msg += '\t total : %s' % len(self.p.keys())
        
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
        varNames = sorted(self.params)
        self.param = [dict(zip(varNames, prod)) for prod in it.product(*(self.params[varName] for varName in varNames))]
        from random import shuffle
        shuffle(self.param)
        self.len = len(self.param)
        self.p = {}
        for i in range(self.len):
            self.p.update({i:self.param[i]})
        self.best_global_mean_err = np.inf
        self.save()
    
    def save(self):
        pickle.dump(self.p,open(self.paramfile,'wb'))
    def load(self):
        if self.verbose: print 'loading' , self.paramfile
        self.p = pickle.load(open(self.paramfile,'rb'))

    def create_todo(self):
        def create_list(self):
            todo = []
            for i, p in zip(self.p.keys(), self.p.values()):
                if p['status'] == 'waiting':
                    todo.append(i)
            return todo 

        while True:
            todo_1 = create_list(self)
            #check times to see if other process wrote to file
            time.sleep(0.3)
            self.load()
            todo_2 = create_list(self)
            if todo_1 == todo_2:
                self.todo = todo_1
                break 
                
    def running(self, id):
        self.p[id]['status'] = 'running'
        self.save()

    def finished(self, id, mean_errs, model):
        self.p[id]['status'] = 'finished'
        self.p[id]['meanerrors'] = mean_errs
        if mean_errs['0-100'] < self.best_global_mean_err:
            self.best_global_mean_err = mean_errs['0-100']
            save_model(model, name='best_ever')
        self.save()
    
    def set_mean_errs(mean_errs):
        #self. = self.set_mean_errs(mean_errs)
        pass

    def print_me(self):
        print '\n complete status'
        print self
        print 'details:'
        for i in self.p.keys():
            print self.p[i]

    def print_par(self, id):
        print 'hyperparameter[%s]\n' % id
        for a, b in zip(self.p[id].keys(), self.p[id].values()):
            print '%s \t%s' %  (a, b)
        print '\n'
#test
#h = Hyperparameter()
#mean_errs = {'0-100':1.5, '0-50':2.0, '50-100':3.0}

