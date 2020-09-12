import numpy as np
import PM_tSNE
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10

X = np.load('./MNIST_data.npy', allow_pickle=True)
y = np.load('./MNIST_labels.npy', allow_pickle=True)
color = ['red','blue','green','orange','black','pink','yellow','brown','purple','grey']
c = np.asarray(color)[y.astype(int)[:]]

tsne = PM_tSNE.PM_tSNE(n_iter=750, coeff=8, grid_meth='NGP')
Y = tsne.fit_transform(X)

plt.scatter(Y[:,0], Y[:,1], c=c[:], s=3)
plt.savefig('./MNIST_NGP_coeff8.png', dpi=400)
