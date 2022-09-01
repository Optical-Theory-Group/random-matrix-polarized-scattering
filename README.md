# Random matrix theory of polarized light scattering in disordered media

---

This code reproduces the data used in *[Random matrix theory of polarized light scattering in disordered media](https://arxiv.org/abs/2205.09423), N. Byrnes and M.R. Foreman, 2022*.

---

## Instructions

This code requires the installation of *[scikit-sparse](https://scikit-sparse.readthedocs.io/en/latest/index.html)* and *[pathos](https://pathos.readthedocs.io/en/latest/index.html)*.

Note: the variable `root` in `main.py` determines the location at which data will be saved. Please set this before running the code!

To generate data, simply run the main.py file. For each parameter set, three files will be generated:
* `data.hdf5`, in which all of the random matrices and polarimetric data is saved
* `params.txt`, which contains some useful information about the simulation parameters
* `statistics.npy`, which contains the mean and covariance matrices associated with the scattering matrix elements

Here we present a brief description of how `main.py` works. The code is best understood with reference to the paper.

1. A grid of modes (transverse wavevectors) is created using `mode_sample_cartesian`. This set of modes determines the size and structure of the scattering matrix. In addition, a list `data_blocks` is defined, which lists the blocks of the scattering matrix for which data is collected.
2. For each set of physical parameters (defined by the dictionaries near the start of the code, e.g. Mie2), the statistics of the scattering matrix elements (mean and covariance) are calculated using `get_statistics`. Specifically, this function returns a dictionary containing the means of the diagonal elements of r, r' and t and the Cholesky decompositions of the correlation matrices associated with the elements of r, r' and t. The Cholesky decompositions are used to generate random (correlated) Gaussian variables. Note that due to their size and structure, the cholesky decompositions are saved as sparse matrices. 
3. Pools of random matrices (as discussed in the paper) are generated using `S_sampler_svd` and saved in `data.hdf5`. Two pools are generated: the 'single pool', which are the building blocks describing very thing slabs, and the 'multi pool', which is a set of matrices describing media of thickness equal to the simulation step size (determined by 'L spacing' in the physical parameter dictionaries). An additional set of matrices used for data collection (referred to here as 'working matrices') is also initialized.
4. The simulation runs through a series of medium thicknesses as defined by the simulation parameters (specifically 'L final' and 'L spacing'). For each new thickness and for each working matrix, a matrix from the multi pool is selected randomly and multiplied to the working matrix. After multiplication, the working matrices then describe different realizations of a scattering medium at the next thickness.
5. For each of the working matrices, various polarimetric quantities are calculated and saved in different sections of `data.hdf5`.
