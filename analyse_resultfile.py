import numpy as np
from sys import exit
import matplotlib
matplotlib.use("GTKAgg")
#matplotlib.use("AGG")
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
import os
basic_path = os.getcwd()

from misc import load_result_file

if __name__ == '__main__':
    path = sys.argv[1]
    data = load_result_file(path)
    print 'hyperpars scanned:%s' % len(data)

    neurons = np.array([int(d['neurons']) for d in data][13:])
    errors = np.array([get_error(d['error']) for d in data][13:])
    batch = np.array([int(d['batch']) for d in data][13:])
    lr = np.array([float(d['lr']) for d in data][13:])

    '''
    mask_ok_err = errors < 9
    plt.scatter(batch[mask_ok_err], errors[mask_ok_err], s=20)
    plt.scatter(neurons[mask_ok_err], errors[mask_ok_err], s=20)
    #plt.scatter(lr[mask_ok_err], errors[mask_ok_err], s=20)
    plt.xlabel('somthing')
    plt.ylabel('error')
    plt.grid()
    plt.show()
    '''
