import sys
import PM_tSNE
import numpy as np
import time

n = int(sys.argv[1])
m = int(sys.argv[2])

X = (np.random.rand(n,m) * 100.0) - 50.0

test = PM_tSNE.PM_tSNE(n_iter=750)

for i in range(1):
    Y = test.fit_transform(X)
