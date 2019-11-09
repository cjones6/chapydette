Chapydette
====================================

Chapydette contains fast Cython implementations of kernel-based change-point detection algorithms and feature generation methods. There are currently two algorithms implemented:

* The kernel change-point algorithm of Harchaoui and Cappé (2007), which detects a fixed number of change points in a sequence of observations.
* The change-point analysis algorithm of Harchaoui, Bach, and Moulines (2009), which detects the presence of and estimates the location of a single change point based on the Maximum Mean Discrepancy (Gretton et al., 2012).

See the full [documentation](http://www.stat.washington.edu/~cjones6/software/chapydette/) for more details. The next release will include the following addition:  
- A data-driven model selection procedure based on Arlot, Celisse, and Harchaoui (2019), which allows one to automatically select the number of change points.

Installation
-----------------
This code was written using Python 3.7 and requires Cython, Faiss, Numpy, PyTorch, Scipy, and Scikit-learn. If you are using Anaconda you can install these via  
```
conda install cython numpy scipy scikit-learn
conda install pytorch torchvision cpuonly -c pytorch
conda install faiss-cpu -c pytorch 
```
If you are using a Mac, you should also install llvm, gcc, and libgcc:  
`conda install llvm gcc libgcc`  
To then install this package, run   
`python setup.py install`   
from the installation directory. 

This code has not been tested on a Windows operating system.

The installer will check whether you have the remainder of the required dependencies. There are two optional dependencies:

* Yael http://yael.gforge.inria.fr/
* Pomegranate https://pomegranate.readthedocs.io/en/latest/index.html

If you have them installed and use them, they may greatly speed up the feature generation code.

Layout
-----------------

There are two tasks that Chapydette can perform:

1. Feature generation
2. Change-point estimation

The functions for the former are contained in feature_generation.py, while the functions for the latter are in cp_estimation.py. Examples of how to use Chapydette are provided in the Jupyter notebook in the examples directory.

Authors
-----------------
[Corinne Jones](https://www.stat.washington.edu/people/cjones6/)  
[Zaid Harchaoui](http://faculty.washington.edu/zaid/)

License
-----------------
This code has a GPLv3 license.

References
-----------------

- S. Arlot, A. Celisse, and Z. Harchaoui, "A kernel multiple change-point algorithm via model selection," *Journal of Machine Learning Research (to appear)*.  
- A. Gretton, K.M. Borgwardt, M.J. Rasch, B. Schölkopf, and A. Smola, "A kernel two-sample test," *Journal of Machine Learning Research*, vol. 13, pp. 723–773, 2012.  
- Z. Harchaoui and O. Cappé, "Retrospective mutiple change-point estimation with kernels," in *IEEE Workshop on Statistical Signal Processing*, 2007, pp. 768–772.  
- Z. Harchaoui, F.R. Bach, and E. Moulines, "Kernel change-point analysis," in *Advances in Neural Information Processing Systems*, 2008, pp. 609–616.
