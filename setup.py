from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    name = 'PM_tSNE',
    ext_modules=[
        Extension('_utils',
                  sources=['_utils.pyx'],
                  extra_compile_args=['-O3'],
                  language='c++')
        ],
    include_dirs=[np.get_include()],
    cmdclass = {'build_ext': build_ext}
)
