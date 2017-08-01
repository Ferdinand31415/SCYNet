import numpy as np
import sys

path = sys.argv[1]
x=np.load(path)
print '\ndata shape:', x.shape

delidx=[]
for i,e in enumerate(x):
    if np.isnan(sum(e)):
        delidx.append(i)

print 'found %s nans\n' % len(delidx)
if len(delidx) != 0:
    y=np.delete(x,delidx,0)
    print 'new data shape:', y.shape
    np.save(path,y)
    print 'saved to path %s' % path
else:
    print 'do nothing'
