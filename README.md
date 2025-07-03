# Active delta-learning for fast construction of interatomic potentials and stable molecular dynamics simulations

The computational details are described at:

- Yaohuang Huang, Yi-Fan Hou, [Pavlo O. Dral](http://dr-dral.com). Active delta-learning for fast construction of interatomic potentials and stable molecular dynamics simulations. Preprint on ChemRxiv: https://doi.org/10.26434/chemrxiv-2024-fb02r.

In the folder `static`, there are all static calculations including geometry optimization, frequency calculations, transition state search, etc. You can visualize the results using the Jupyter notebook there.

In the folder `adl`, there are scripts with active delta learning codes. In the subfolder `example_scripts`, you will find example scripts to run ADL. In subfolders start with `ADL`, you can find the ADL calculations (including scripts, model and training set) shown in the paper. In the `analyzing_codes` subfolder, you can find our scripts that analyze trajectories.