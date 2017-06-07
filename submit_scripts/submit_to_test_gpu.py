import subprocess as sp
import time, sys

class GPU_MASTER():
    def __init__(self):
        self.start_time = time.time()

    def print_time(self):
        '''prints time passed since start'''
        t = time.time() - self.start_time
        t = int(t)
        d = t / 86400
        t = t % 86400
        h = t / 3600
        t = t % 3600
        print '\ncurrent duration: %sd %sh %smin-%ss\n' % (d, h, t/60, t%60)
    
    def get_info(self, cmd='nvidia-smi',verbose=True):
        info= sp.Popen(cmd,stdout=sp.PIPE,shell=True)
        self.result = info.communicate()[0].split('\n')
        self.success = not bool(info.returncode)
        if verbose:
            self.print_info()

    def gpu_free(self,verbose=True):
        if self.success:
            for line in self.result:
                if 'No running processes found' in line:
                    return True
            return False

    def print_info(self):
        for i in self.result:
            print i

    def run_testgpu(self):
        cmd = 'python ../randhyperscan_singleepochs.py'
        info = sp.Popen(cmd,shell=True,stdout=sp.PIPE)
        res = info.communicate()
        
        print '\n\nreturncode randhyperscan_singleepochs.py %s'%info.returncode
        for i in res[0].split('\n'):
            print i

        return info.returncode

if __name__ == '__main__':
    gpu = GPU_MASTER()
    i = 0
    while i < 10:
        i += 1
        print '\n%s\nITERATION %s\n' % (50*'_', i)
        gpu.print_time()
        gpu.get_info()
        if gpu.gpu_free():
            exit_code = gpu.run_testgpu()
            if exit_code != 0:
                sys.exit('??')

        wait = 5*60
        print '\nsleeping %s secs..' % wait
        time.sleep(wait)
    
