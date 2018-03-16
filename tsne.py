from dstML.data_visualization import *
from dstML.data_fn import *
import numpy as np
from scipy.io import loadmat

if __name__ == "__main__":
	
	#X = np.loadtxt("data/mnist/mnist2500_X.txt")[:500,:]
	#labels = np.loadtxt("data/mnist/mnist2500_labels.txt")[:500]
	# load data
	data_dict = loadmat('data/oct_data.mat')
	X,labels = data_dict['data'],data_dict['label'][0,:]
	X = normalize(X)
	#X = (X-X.min())/(X.max()-X.min())
	print X.min(),X.max()
	Y = tsne(X, 2, 23,10,200)
	Plot.scatter(Y[:,0], Y[:,1], 20, labels)
	Plot.show()
