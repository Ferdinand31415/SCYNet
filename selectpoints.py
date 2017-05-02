from netbuilder import load_model
from Preprocessor import pmssm
import sys

'''script that takes a model and output from preselection scan.
We then pass this with a range to the model and get new points for the
Eventgeneration Chain'''

modname = sys.argv[1]
model = load_model(name = modname) #needs modname.json and modname.h5 files

path_to_points = sys.argv[2]
points = np.genfromtxt(path_to_points)

upper, lower = sys.argv[3], sys.argv[4]
preproc = ['sub_mean_div_std'] #get this how? #read from hyperrandomscan file..?
x = pmssm(points, preproc, split = 1.0)
chi2_pred = model.predict(x)
mask = (chi2_pred > lower) == (chi2 < upper)
chi2_pred = chi2_pred[mask]


result_file = sys.argv[5]
with open(result_file, 'w') as file:#destroys any existing result_file
    for i in range(len(chi2_pred)):
        file.write(' '.join(map(str, x[0])) + '\n')

