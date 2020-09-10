import sys
import PM_tSNE
import numpy as np
import time
from tqdm import tqdm

n = int(sys.argv[1])
m = int(sys.argv[2])
coeff = float(sys.argv[3])

X = (np.random.rand(n,m) * 100.0) - 50.0

test = PM_tSNE.PM_tSNE(n_iter=750, coeff=coeff)

start = time.perf_counter()
for i in tqdm(range(10)):
    Y = test.fit_transform(X)
print('New code :', time.perf_counter() - start)
