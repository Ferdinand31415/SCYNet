import os
import numpy as np
from copy import deepcopy

from sklearn.utils import shuffle
def shuffle_data(x, y):
    x.x, y.chi2 = shuffle(x.x, y.chi2) #option: random_state=0 controlling seed
    y.train_test_init_sq() #so we compare prediction with correct ytrain,test
    x.split_train_data() #new train split
    y.split_train_data() #new test split

class fulldata:
    check = False
    def __init__(self, path = os.environ['HOME'] + '/13TeV_chi2_disjoint_2.npy',use=range(0,12)):
        self.path = path
        #self.data = np.genfromtxt(self.path[:-4])[:,1:]
        self.data = np.load(self.path)[:,use]
        #print 'loaded pmssm11 data with shape', self.data.shape
        if self.check==True:
            self.quality_ensurance()

    def shuffle(self,seed=None):
        if seed != None:
            np.random.seed(seed)
        np.random.shuffle(self.data)
    def quality_ensurance(self):
        print 'checking if a "nan" is in the data....'
        if np.isnan(sum(sum(self.data))):
            raise ValueError('found a "nan" in the input data')
        else:
            self.check = False

class pmssm:
    def __init__(self, data, preproc, split, use_only=range(11)):
        self.use_only = use_only
        self.x = data
        self.split = int(split*len(self.x))
        self.preproc = preproc
        self.back_info = {}
        self.preprocess() 

    def sub_mean_div_std(self):
        self.mod = np.zeros((11,2))
        backinfo = []
        for i in range(self.x.shape[1]):
            mean = self.x[:,i].mean()
            std = self.x[:,i].std()
            self.x[:,i] = (self.x[:,i] - mean) / std
            self.mod[i] = [mean, std]
            backinfo.append([mean, std])
        self.back_info.update({'sub_mean_div_std':backinfo})

    def div_max(self):
        '''divide by maximum'''
        self.maxis = np.zeros(11)
        for i in range(self.x.shape[1]):
            maxi = max(abs(self.x[:,i]))
            self.x[:,i] /= maxi 
            self.maxis[i] = maxi
        self.back_info.update({'div_max':list(self.maxis)})

    def min_max(self):
        '''normalize between zero and one'''
        self.minmaxis = np.zeros((11,2))
        mimaxis = []
        for i in range(self.x.shape[1]):
            mini = min(self.x[:,i])
            maxi = max(self.x[:,i])
            self.minmaxis[i] = [mini, maxi]
            mimaxis.append([mini,maxi])
            self.x[:,i] = (self.x[:,i] - mini) / (maxi - mini)
        self.back_info.update({'min_max':mimaxis})        

    def log_norm(self):
        '''scale by log'''
        self.logis = np.zeros(11)
        for i in range(self.x.shape[1]):
            p = self.x[:,i]
            mini = min(p)
            mask = p < 0
            p[mask] *= -1
            p = np.log(p - mini + 2) #+2 regularizes, we want no log(0)
            self.logis[i] = mini
            #print 'logis', i, self.logis[i], mini, p[:10]
            p[mask] *= -1
            self.x[:,i] = p
        self.back_info.update({'log_norm':list(self.logis)})

    def split_train_data(self):
        '''simple split function'''
        use_only = self.use_only #never does anything. If you want to use only certain pmssm parameters, you may take advantage of this.
        self.train = self.x[:,use_only][:self.split]
        self.test = self.x[:,use_only][self.split:]

    def preprocess(self):
        for p in self.preproc:
            getattr(self, p)() #retrieves function name, calls it with ()
        self.split_train_data()

class chi2:
    '''reads in the chi2, has functionality for evaluation of model'''
    def __init__(self, data, preproc, params, split=6.0/8, verbose=False):
        self.verbose = verbose #just for some printing
        self.preproc = preproc
        self.params = params
        self.chi2 = data
        self.split = int(split*len(self.chi2))
        self.start = deepcopy(self.chi2[:50])
        self.back = [] #training+SCYNet
        self.back_info = {}#not needed during training. but for SCYNet.
        self.preprocess() 

    def split_train_data(self):
        self.train = self.chi2[:self.split]
        self.test = self.chi2[self.split:]
 
    '''divide chi2 by maximum'''
    def div_max(self):
        self.maximum = max(self.chi2)
        self.chi2 /= self.maximum
        self.back.append('mult_max')#training + SCYNet
        self.back_info.update({'maximum':self.maximum})#used only for SCYNet
    def mult_max(self, chi2):
        return self.maximum * chi2

    '''substract mean and divide by std'''
    def sub_mean_div_std(self):
        self.mean = self.chi2.mean()
        self.std = self.chi2.std()
        self.chi2 = (self.chi2 - self.mean)/self.std
        self.back.append('mult_std_add_mean')
        self.back_info.update({'mean':self.mean, 'std':self.std})
    def mult_std_add_mean(self, chi2):
        return self.std*chi2 + self.mean

    '''transform arbitrary thing to 0..1'''
    def min_max(self):
        self.mini = min(self.chi2)
        self.maxi = max(self.chi2)
        self.chi2 = (self.chi2 - self.mini)/(self.maxi - self.mini)
        self.back.append('max_min')
        self.back_info.update({'mini':self.mini, 'maxi':self.maxi})
    def max_min(self, chi2):
        return (self.maxi - self.mini)*chi2 + self.mini

    def hard_cut(self):
        cut = self.params[0]
        self.chi2[self.chi2 > cut] = cut

    '''smoothly cut chi2 data at self.cut'''
    def square_cut(self):
        cut, delta = self.params
        y = self.chi2
        self.mask = (y > (cut - delta)) == (y < (cut + delta)) #true for max-cut <= chi2 <= max+cut, false else
        y[self.mask] = cut - (y[self.mask] - cut - delta)**2/(4*delta)
        y[y > (cut + delta)] = cut
        self.chi2 = y   
        self.back.append('back_square')
        self.back_info.update({'cut':cut, 'delta':delta})
    def back_square(self, chi2):
        cut, delta = self.params
        self.cut = cut
        self.delta = delta
        y = chi2
        mask = (y > cut-delta) == (y < cut) #backtrafo needs a different mask than trafo
        y[mask]= (cut+delta)-2*np.sqrt(delta*(cut-y[mask]))
        y[y>cut] = cut
        return y

    
    def train_test_init_sq(self):
        #only works for square cut! #if tanh, add another!#TODO
        self.cut = self.params[0]
        self.train_init = deepcopy(self.chi2)[:self.split].flatten()
        self.train_init[self.train_init > self.cut] = self.cut

        self.test_init= deepcopy(self.chi2)[self.split:].flatten()
        self.test_init[self.test_init > self.cut] = self.cut

    def preprocess(self):
        if True: #if I implement tanh cut, change this#TODO
            self.train_test_init_sq()
        for p in self.preproc:
            prefunc = getattr(self, p)
            prefunc()
        self.split_train_data()   

    def backtrafo(self, chi2):
        for func in reversed(self.back): #apply in reverse order
            backfunc = getattr(self, func)
            chi2 = backfunc(chi2)
        return chi2

    def prepare_evaluation(self, x, model):
        #predictions
        if self.verbose: print 'preparing_evaluations'
        y_train_pred = model.predict(x.train)
        y_test_pred = model.predict(x.test)
        if self.verbose: print y_test_pred.flatten()[:20]
        self.train_pred = self.backtrafo(y_train_pred.flatten())
        self.test_pred = self.backtrafo(y_test_pred.flatten())
        if self.verbose: print self.test_pred[:20]

    def evaluation(self,x,model):
        self.prepare_evaluation(x, model) 
        ranges = [[0,y.cut], [0, 53.5], [53.5, 56.0],\
                 [56.0, 70.0], [70.0, 95.0], [95.0, y.cut]]
        l=11
        chi2 =  'chi2              '
        deco =  '                  '+l*len(ranges)*'_'
        train = 'mean error train |'
        test =  'mean error test  |'

        ws =4*' '
        result_fancy ='chi2        test   train';L=len(result_fancy)
        result_fancy+='\n' + L*'-'+'\n'
        self.mean_errors = {}
        self.nan_error = False #if a value we calculate is 'nan' we set it to True
        for iter_r,r in enumerate(ranges):
                if self.verbose: print r
                mask_train = (self.train_init>r[0]) == (self.train_init<=r[1])
                mask_test = (self.test_init>r[0]) == (self.test_init<=r[1])
                chi2_mod = '%s-%s' % ('{0:.1f}'.format(r[0]), '{0:.1f}'.format(r[1]))
                err_train= np.mean(np.abs(self.train_pred[mask_train]-self.train_init[mask_train]))
                err_test = np.mean(np.abs(self.test_pred[mask_test]-self.test_init[mask_test]))
                train_mod = "{0:.2f}".format(err_train)
                test_mod = "{0:.2f}".format(err_test)
                #train_mod =str(err_train)[:4]
                #test_mod =str(err_test)[:4]
                self.mean_errors[chi2_mod] = [err_train,err_test]
                chi2 += chi2_mod + (l-len(chi2_mod))*' '
                train += train_mod + (l-len(train_mod))*' '
                test += test_mod +(l-len(test_mod))*' '
                result_fancy += chi2_mod + (12-len(chi2_mod))*' '+test_mod+'   '+train_mod+'\n'
                if iter_r == 0: result_fancy += L*'-'+'\n'
                if np.isnan(err_train) or np.isnan(err_test):
                    self.nan_error = True
                    print 'WARNING: got nan: err train:%s ,err test: %s' % (err_train,err_test) 
        self.err = self.mean_errors['0.0-100.0'][1] 
        #print '\n' + chi2
        #print deco
        #print train
        #print test + '\n'
        print result_fancy

    def plot_chi2_mean_error(self,y_train_pred, y_test_pred, y_train, y_test,bins=30):
        bins += 1
        mini = min(min(y_train),min(y_test))
        maxi = max(max(y_train),max(y_test))
        
        ZS_N, ZS_Area = 8, 5
        if bins <= ZS_N:
            print 'make more bins..'
            return 
        ZS = np.linspace(mini, mini + ZS_Area, ZS_N + 1)[:-1]
        RT = np.linspace(mini + ZS_Area, maxi, bins-ZS_N)
        r = np.concatenate([ZS,RT], axis=0)
        #print ZS
        #print RT
        #print r
        mean_err_train, mean_err_test = np.zeros(len(r)), np.zeros(len(r))
        #print mean_err_train.shape, r.shape
        for i in range(len(r)-1):
            mask_train = (y_train>r[i]) == (y_train<=r[i+1])
            mask_test = (y_test>r[i]) == (y_test<=r[i+1])
            if np.all(mask_train==0) or np.all(mask_test==0):	
                continue
            err = np.mean(np.abs(y_train_pred[mask_train]-y_train[mask_train]))
            mean_err_train[i] = 0 if np.isnan(err) else err
            err = np.mean(np.abs(y_test_pred[mask_test]-y_test[mask_test]))
            mean_err_test[i] = 0 if np.isnan(err) else err
        plt.plot(r, mean_err_train,drawstyle='steps-post',color='red')
        plt.plot(r, mean_err_test,drawstyle='steps-post',color='green')
        plt.xlim(mini,maxi) #min,max of r
        plt.xlabel('chi2')
        plt.ylabel('mean error')
        plt.grid()
        plt.title('mean errors for net')
        #plt.show()
        plt.savefig('./output/mean_errors_chi2')
        plt.gcf().clear()

    def save(self):
        np.save('./output/y_train_pred',self.train_pred)
        np.save('./output/y_train_init',self.train_init)
        np.save('./output/y_test_pred',self.test_pred)
        np.save('./output/y_test_init',self.test_init)
 
###############################
# preprocessing for SCYNet.py #
###############################
def sub_mean_div_std(data, smds):
    for i in range(data.shape[1]):
        mean, std = smds[i]
        data[:,i] = (data[:,i] - mean)/std
    return data
def log_norm(data, logis):
    for i in range(data.shape[1]):
        p = data[:,i]
        mask = p < 0
        p[mask] *= -1
        p= np.log(p - logis[i] + 2)
        p[mask] *= -1
        data[:,i] = p
    return data
def div_max(data, maxis):
    for i in range(data.shape[1]):
        data[:,i] /= maxis[i]
    return data
def min_max(data, minmaxis):
    for i in range(data.shape[1]):
        mini, maxi = minmaxis[i]
        data[:,i] = (data[:,i] - mini)/(maxi - mini)
    return data


if __name__ == '__main__': 
    y = chi2(['square_cut','div_max'], [100,25], split=7.0/8)

    back = y.backtrafo(y.train)
    print y.start

    for a,b,c in zip(y.train[:50], y.train_init[:50], back[:50]):
        print b,c,c==b#==c
    #x=pmssm(preproc = ['log_norm','min_max'], split = 7.0/8)
    #y=pmssm(preproc = ['sub_mean_div_std','min_max'], split = 7.0/8)

    #import matplotlib
    #matplotlib.use("GTKAgg")
    #import matplotlib.pyplot as plt

    #print x.x[:,0]
    #print y.x[:,0]

    #plt.hist(y.x[:,0], bins=100)
    #plt.show()
    
