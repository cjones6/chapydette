Chapydette
====================================

Chapydette contains fast Cython implementations of kernel-based change-point detection algorithms and feature generation methods. There are currently two main algorithms implemented:

* The kernel change-point algorithm of Harchaoui and Cappé (2007), which detects a fixed number of change points in a sequence of observations. We also provide an implementation of the data-driven model selection procedure based on Arlot, Celisse, and Harchaoui (2019), which allows one to automatically select the number of change points.
* The change-point analysis algorithm of Harchaoui, Bach, and Moulines (2009), which detects the presence of and estimates the location of a single change point based on the Maximum Mean Discrepancy (Gretton et al., 2012).

Installation
-----------------
This code was written using Python 3.7 and requires Cython, Faiss, Jupyter, Matplotlib, Numba, Numpy, PyTorch, Scipy, and Scikit-learn. If you are using Anaconda you can install these in a new conda environment called `chpt` via  
```
conda create -y --name=chpt python=3.7
conda activate chpt
conda install cython jupyter matplotlib nb_conda numba numpy scipy scikit-learn
conda install pytorch torchvision cpuonly -c pytorch
conda install faiss-cpu -c pytorch 
```

In order to compile the code you will need to have gcc installed. If you are using Ubuntu, you can install this via  
`sudo apt install build-essential`  
If you are using a Mac, you should install llvm, gcc, and libgcc:   
`conda install llvm gcc libgcc`  

To then install this package, run   
`python setup.py install`   
from the installation directory. 

This code has not been tested on a Windows operating system.

The installer will check whether you have the remainder of the required dependencies. There are two optional dependencies:

* Yael http://yael.gforge.inria.fr/
* Pomegranate https://pomegranate.readthedocs.io/en/latest/index.html

If you have them installed and use them, they can greatly speed up the feature generation code.

Layout
-----------------

There are two tasks that Chapydette can perform:

1. Feature generation
2. Change-point estimation

The functions for the former are contained in feature_generation.py, while the functions for the latter are in cp_estimation.py. Examples of how to use Chapydette are provided in the Jupyter notebooks in the examples directory.

Authors
-----------------
[Corinne Jones](https://www.stat.washington.edu/people/cjones6/)  
[Zaid Harchaoui](http://faculty.washington.edu/zaid/)

Citation
-----------------
If you use this code in your work please cite one of the following papers:

- C. Jones and Z. Harchaoui. [End-to-End Learning for Retrospective Change-Point Estimation](https://ieeexplore.ieee.org/document/9231768). In *Proceedings of the IEEE International Workshop on Machine Learning for Signal Processing*, 2020.

```
@inproceedings{JH2020,
  author    = {Jones, Corinne and Harchaoui, Zaid},
  title     = {End-to-End Learning for Retrospective Change-Point Estimation},
  booktitle = {30th {IEEE} International Workshop on Machine Learning for Signal Processing},
  year      = {2020},
  doi       = {10.1109/MLSP49062.2020.9231768}
}
```

- C. Jones, S. Clayton, F. Ribalet, E.V. Armbrust, and Z. Harchaoui. [A Kernel-Based Change Detection Method to Map Shifts in Phytoplankton Communities Measured by Flow Cytometry](https://www.biorxiv.org/content/early/2020/12/02/2020.12.01.405126). bioRxiv, 2020. 
```
@article {JCRAH2020,
	author       = {Jones, Corinne and Clayton, Sophie and Ribalet, Fran{\c c}ois and Armbrust, E. Virginia and Harchaoui, Zaid},
	title        = {A Kernel-Based Change Detection Method to Map Shifts in Phytoplankton Communities Measured by Flow Cytometry},
	elocation-id = {2020.12.01.405126},
	year         = {2020},
	doi          = {10.1101/2020.12.01.405126},
	publisher    = {Cold Spring Harbor Laboratory},
	journal      = {bioRxiv}
}
```


License
-----------------
This code has a GPLv3 license.

References
-----------------

- S. Arlot, A. Celisse, and Z. Harchaoui, "A kernel multiple change-point algorithm via model selection," *Journal of Machine Learning Research*, 2019.  
- A. Gretton, K.M. Borgwardt, M.J. Rasch, B. Schölkopf, and A. Smola, "A kernel two-sample test," *Journal of Machine Learning Research*, vol. 13, pp. 723–773, 2012.  
- Z. Harchaoui and O. Cappé, "Retrospective mutiple change-point estimation with kernels," in *IEEE Workshop on Statistical Signal Processing*, 2007, pp. 768–772.  
- Z. Harchaoui, F.R. Bach, and E. Moulines, "Kernel change-point analysis," in *Advances in Neural Information Processing Systems*, 2008, pp. 609–616.
