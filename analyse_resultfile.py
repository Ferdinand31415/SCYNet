import numpy as np

import matplotlib
matplotlib.use("GTKAgg")
#matplotlib.use("AGG")
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
import os, sys
basic_path = os.getcwd()

#from misc import load_result_file
def result_string_to_dict(line,verbose=False):
    hp = {}
    L = line.split(';')
    for l in L:
        key, value = l.split('?')
        if verbose: print key, '\t', value
        hp.update({str(key):str(value)})
        try:
            hp[key] = eval(hp[key])
        except NameError:
            pass
    return hp

def load_result_file(path):
    data = []
    with open(path,'r') as file:
        lines = file.readlines()
    for l in lines:
        data.append(result_string_to_dict(l,verbose=False))
    return data

def scatter(x, y, label = 'something'):
    plt.gcf().clear()
    plt.scatter(x, y, s=20)
    plt.xlabel(label)
    plt.ylabel('mae -- mean average error')
    plt.grid()
    if len(sys.argv)>2 and sys.argv[2] == 'plt':
        plt.show()

if __name__ == '__main__':
    path = sys.argv[1]
    data = load_result_file(path)
    L = len(data)
    print 'hyperpars scanned:%s' % L

    neurons = np.array([d['neurons'] for d in data])
    errors = np.array([d['error'] for d in data])
    batch = np.array([int(d['batch']) for d in data])
    lr = np.array([float(d['lr']) for d in data])
    dropout = np.array([d['dropout'] for d in data])
    beta_1 = np.array([d['beta_1'] for d in data])
    beta_2 = np.array([d['beta_2'] for d in data])
    layers = np.array([d['layers'] for d in data])
    nadam = np.array([True if d['opt'] == 'nadam' else False for d in data])
    sgd = np.invert(nadam)

    #preprocessing
    chi2_min_max = np.zeros(L, dtype='bool')
    chi2_sub_mean_div_std = np.zeros(L, dtype='bool')
    chi2_div_max = np.zeros(L, dtype='bool')

    for i in range(L):
        pp = data[i]['pp_chi2'][1]
        if pp == 'min_max':
            chi2_min_max[i] = True
        if pp == 'sub_mean_div_std':
            chi2_sub_mean_div_std[i] = True
        if pp == 'div_max':
            chi2_div_max[i] = True
            
    #only working nets
    mask_ok_err = errors < 9.9 
    #compact or rising/falling network?
    c = np.array([True if n==n[::-1] else False for n in neurons])
    nc = np.invert(c)

    neurons_c = np.array([n[0] for n in neurons[c]])
    
    eok = errors[mask_ok_err]
    cok = c[mask_ok_err]
    print 'mean errors compact/noncompact'
    print np.mean(errors[c]), np.mean(errors[np.invert(c)])
    print np.mean(eok[cok]), np.mean(eok[np.invert(cok)]), '\n'

    for i in range(2,7):
        layermask = layers == i
        print np.mean(errors[layermask]), i
   
    print 'mean errors pp_chi2: min_max, sub_mean_div_max, div_max'
    pp_eok_mm = np.logical_and(chi2_min_max, mask_ok_err)
    pp_eok_smds = np.logical_and(chi2_sub_mean_div_std, mask_ok_err)
    pp_eok_dv = np.logical_and(chi2_div_max, mask_ok_err)
    
    print np.mean(errors[pp_eok_mm]), np.mean(errors[pp_eok_smds]), np.mean(errors[pp_eok_dv])
    print len(errors[pp_eok_mm]), len(errors[pp_eok_smds]), len(errors[pp_eok_dv])

    print np.mean(errors[chi2_sub_mean_div_std])

    #time analysis.
    times_to_check = [0.045, 0.03, 0.05, 0.035, 0.02, 0.055, 0.04, 0.025, 0.06]
    times = {key: [] for key in times_to_check}
    for d in data:
        try:
            for time, error in d['times'].iteritems():
                times[time].append(error)
        except:
            pass
    print 'stats timing:'
    for t in sorted(times.keys()):
        print '\ttime:%s\tentries:%s\tmean time: %s' % (t, len(times[t]), sum(times[t])/len(times[t]))

