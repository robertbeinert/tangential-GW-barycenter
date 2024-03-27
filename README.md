# Tangential Gromov-Wasserstein Barycenters

This repository contains the code for the paper 'Tangential Fixpoint Iterations for Gromov-Wasserstein Barycenters'. 
A preprint version is available on [arXiv](https://arxiv.org/abs/2403.08612).

Please cite the paper if you use the code.

## Citation
1. Florian Beier, Robert Beinert
    'Tangential Fixpoint Iterations for Gromov-Wasserstein Barycenters'',
    arXiv:2403.08612.

## Requirements
The simulations have been performed with Python 3.8.8 and rely on 

* numpy 1.24.2,
* scipy 1.10.0,
* matplotlib 3.7.1,
* pot 0.8.0,
* tqdm 4.62.3,
* trimesh 3.10.5,
* scikit-learn 1.0.1,
* networkx 2.6.3,
* open3d 0.17.0,
* GromovWassersteinFramework.py from [S-GWL](https://github.com/HongtengXu/s-gwl>).

## Experiments
The numerical simulations can be reproduced using the script

* `Ex1-3d-Interpolation.ipynb` 
    for the 3d shape interpolations between 2 and 4 input shapes (Figs 1, 2),

* `Ex2-LGW-DF.ipynb` 
    for the classifying the 3d registrations from the mesh deformation dataset (Fig 3, Tab. 1),

* `Ex2-LGW-FAUST.ipynb` 
    for the classifying the 3d meshes from the FAUST dataset (Fig 4, Tab. 2),

* `Ex3-PPI.ipynb` 
    for simultaneously matching multiple PPI networks (Tab. 3).

Parts of the implementation relies on or is built on top of existing implementations from 
[Python Optimal Transport](https://pythonot.github.io/)
and
[S-GWL](https://github.com/HongtengXu/s-gwl>).

The input data is based on
[the 3D Mesh Deformation Dataset](http://people.csail.mit.edu/sumner/research/deftransfer/data.html) and
[the FAUST Dataset](https://faust-leaderboard.is.tuebingen.mpg.de/).

## Contributing
The code is available under a MIT license.
