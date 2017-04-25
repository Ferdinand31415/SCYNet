import itertools as it
import sys
import numpy as np
from numpy.random import RandomState

try:
    import cPickle as pickle
except:
    import pickle


class hyperparameter():
    
    def __init__(self):
        self.paramfile = 'paramfile.pkl'
        self.scannedfile = 'alrdyscanned'#why did i create this?
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
        layers =[[50,50],[100,100,100],[500,500,500],[300,300,300]]
        return layers

    def init(self):
        '''only do this once to initialize the paramaters.'''
        self.params = {}
        self.params['optimizer']=['nadam']
        #self.params['lr']=[0.1,0.01,0.001] #do this one interactively while training. not needed here
        self.params['layers']= self.create_layers()
        self.params['activation']=['relu']
        self.params['status'] = ['waiting']
        #self.params['regularization'] = ['l2','dropout']
        self.params['dropout'] = [0.01, 0.05, 0.1, 0.2, 0.4]
        self.params['meanerrors'] = ['tbd']
        varNames = sorted(self.params)
        self.param = [dict(zip(varNames, prod)) for prod in it.product(*(self.params[varName] for varName in varNames))]
        from random import shuffle
        shuffle(self.param)
        self.len = len(self.param)
        self.p = {}
        for i in range(self.len):
            self.p.update({i:self.param[i]})
        self.save()
    
    def save(self):
        pickle.dump(self.p,open(self.paramfile,'wb'))
    def load(self):
        self.p = pickle.load(open(self.paramfile,'rb'))

    def create_todo(self):
        self.todo = []
        for i, p in zip(self.p.keys(), self.p.values()):
            if p['status'] == 'waiting':
                self.todo.append(i)
                
    def running(self, id):
        self.p[id]['status'] = 'running'
        self.save()

    def finished(self, id):
        self.p[id]['status'] = 'finished'
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

#test
h = Hyperparameter()
mean_errs = {'0-100':1.5, '0-50':2.0, '50-100':3.0}

