from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np

extensions = [
    Extension(
        "chapydette/*.pyx",
        ["chapydette/*.pyx"],
        include_dirs=['.', 'chapydette',  np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

setup(name='Chapydette',
      version='0.1',
      description='Kernel-based changepoint detection methods',
      long_description='',
      author='Corinne Jones with Zaid Harchaoui',
      author_email='cjones6@uw.edu',
      packages=find_packages(),
      package_data={
          'chapydette': ["chapydette/*.pxd"],
      },
      ext_modules=cythonize(extensions),
      include_dirs=['.', 'chapydette', np.get_include()],
      install_requires=[
          'Cython',
          'numpy',
          'scipy',
          'scikit-learn',
          ],
      )