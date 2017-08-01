import subprocess as sp
import time, sys, os
from unipath import Path

class GPU_MASTER():
    def __init__(self,verbose=False):
        self.verbose = verbose
        self.start_time = time.time()
        self.basic_path = str(Path(os.getcwd()).parent)
        print 'basic_path',self.basic_path

    def print_time(self):
        '''prints time passed since start'''
        t = time.time() - self.start_time
        t = int(t)
        d = t / 86400
        t = t % 86400
        h = t / 3600
        t = t % 3600
        print '\ncurrent duration: %sd %sh %smin-%ss\n' % (d, h, t/60, t%60)
    
    def get_info(self, cmd='nvidia-smi'):
        info= sp.Popen(cmd,stdout=sp.PIPE,shell=True)
        self.result = info.communicate()[0].split('\n')
        self.success = not bool(info.returncode)
        if self.verbose:
            self.print_info()

    def check_gpu_free(self):
        if self.success:
            for line in self.result:
                if 'No running processes found' in line:
                    return True
                if 'C   python' in line:#thats me... Im not blocking myself
                    return True
        return False

    def gpu_free(self):
        self.get_info()
        return self.check_gpu_free()
 
    def print_info(self):
        for i in self.result:
            print i

    def run_testgpu(self,arg):
        cmd = 'python randhyperscan_singleepochs.py %s' % arg
        info = sp.Popen(cmd,shell=True,stdout=sp.PIPE,cwd = self.basic_path)
        res = info.communicate()
        
        print '\n\nreturncode randomhyperscan.py %s'%info.returncode
        for i in res[0].split('\n'):
            print i
    
        print info
        print res
        return info.returncode

if __name__ == '__main__':
    pass
