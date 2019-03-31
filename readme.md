Chapydette
====================================

Chapydette contains fast Cython implementations of kernel-based change-point detection algorithms and feature generation methods. There are currently two algorithms implemented:

* The kernel change-point algorithm of Harchaoui and Cappé (2007), which detects a fixed number of change-points in a sequence of observations.
* A change-point algorithm based on the Maximum Mean Discrepancy (Gretton et al., 2012) to detect a single change-point. This algorithm is based on Harchaoui et al. (2009).

See the full [documentation](http://www.stat.washington.edu/~cjones6/software/chapydette/) for more details.

Installation
-----------------

To install the package, run `python setup.py install` from the installation directory, with `sudo` if necessary. This code was written using Python 2.7 and requires Cython. It is not compatible with Python 3 and has been only tested on 64-bit Linux. 

The installer will check whether you have the remainder of the required dependencies. There are three optional dependencies:

* Faiss https://github.com/facebookresearch/faiss
* Yael http://yael.gforge.inria.fr/
* Pomegranate https://pomegranate.readthedocs.io/en/latest/index.html

If you have them installed and use them, they will greatly speed up the feature generation code.

Layout
-----------------

There are two tasks that Chapydette can perform:

1. Feature generation
2. Change-point estimation

The functions for the former are contained in feature_generation.py, while the functions for the latter are in cp_estimation.py. Examples of how to use Chapydette are provided in the ipython notebook in the examples directory.

Author
-----------------
[Corinne Jones](https://www.stat.washington.edu/people/cjones6/)  

License
-----------------
This code has a GPLv3 license.

References
-----------------

- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. Journal of Machine Learning Research, 13(Mar), 723-773.  
- Harchaoui, Z., & Cappé, O. (2007, August). Retrospective multiple change-point estimation with kernels. In IEEE/SP 14th Workshop on Statistical Signal Processing, 2007. SSP'07. (pp. 768-772). IEEE.  
- Harchaoui, Z., Moulines, E., & Bach, F. R. (2009). Kernel change-point analysis. In Advances in Neural Information Processing Systems (pp. 609-616).