# PM_tSNE_v2

### Description

This code implement a Particle-Mesh inspired t-SNE code.

The Particle-Mesh algorithm (PM) is well used in physics to solve
the N-body problem. It is based on the ability of FFTs to transform
convolution products in simple products. The main idea is to compute
a potential, and get back the forces by deriving it.

### Dependencies

Python 3.6.9 was used during the development of this code.

Check the 'requirements' file to install dependencies.

* Annoy 1.16.3
* Cython 0.29.21
* Numpy 1.19.2
* Sklearn 0.23.2
* Scipy 1.5.2

### Installation :

```python
>> python setup.py build_ext --inplace
```

### Verification :

If matplotlib is installed, you can try...

```python
>> python test_PM.py
```

... in order to test the implementation on the 70 000 instances of MNIST.

### Utilisation :

```python
import PM_tSNE
tsne = PM_tSNE.PM_tSNE(n_iter=750, coeff=8.0, grid_meth='NGP')
# Load the 70 000 instances of MNIST (already prepared for t-SNE: standardized + reduced to 50 features with PCA)
X = np.load('./MNIST_data.npy', allow_pickle=True)
Embedding = tsne.fit_transform(X)
```
