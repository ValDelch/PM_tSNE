# PM_tSNE_v2

Implementation of a Particle-Mesh inspired t-SNE algorithm.

Python 3.6.9 was used during the development of this code.
Check the 'requirements' file to install dependencies.
	* Annoy 1.16.3
	* Cython 0.29.21
	* Numpy 1.19.2
	* Sklearn 0.23.2
	* Scipy 1.5.2

Installation :

>> python setup.py build_ext --inplace

Verification :

If matplotlib is installed in addition to dependencies, you can try...

>> python test_PM.py

... in order to test the implementation on the 70 000 instances of MNIST.
