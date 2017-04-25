import numpy as np
# represents output of last layer (activation at last layer)
a = np.random.randn(5)
expa = np.exp(a)
softmaxa = expa/expa.sum()

# now with matrix
A = np.random.randn(100,5)
expA = np.exp(A)
softmaxA = expA/expA.sum(axis=1, keepdims=True) #sum each row
