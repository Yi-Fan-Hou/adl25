import mlatom as ml

# dir = '/export/home/huangyh/projects/pve_2/tri/ADL/crest/lll6/min'
mol = ml.data.molecule.load('b3_r.xyz',format='xyz')

b3lyp = ml.models.methods(method='B3LYP/6-31G*',program='Gaussian',nthreads=12,save_files_in_current_directory=False)
# d4 = ml.models.methods(method='D4', functional='b3lyp',save_files_in_current_directory=False,nthreads=4)
# b3lyp_model_tree = ml.models.model_tree_node(name='b3lyp',operator='predict',model=b3lyp)
# d4_model_tree = ml.models.model_tree_node(name='d4',operator='predict',model=d4)
# method = ml.models.model_tree_node(name='ub3lyp_d4',children=[b3lyp_model_tree,d4_model_tree],operator='sum')

geomopt = ml.optimize_geometry(ts=False,
                               model=b3lyp,
                               initial_molecule=mol,
                               program='gaussian',
                               program_kwargs={'opt_keywords': ['calcall', 'nomicro']}
                               )

# Get the final geometry approximately at the full CI level
final_mol = geomopt.optimized_molecule
print('Optimized coordinates:')
print(final_mol.get_xyz_string())
final_mol.write_file_with_xyz_coordinates(filename='final_r.xyz')

# Let's check how many full CI calculations, our delta-model saved us
print('Number of optimization steps:', len(geomopt.optimization_trajectory.steps))

# Calculate frequency
freq = ml.freq(model=b3lyp, molecule=final_mol, program='gaussian')

# Check vibration analysis
print("Mode     Frequencies     Reduced masses     Force Constants")
print("           (cm^-1)            (AMU)           (mDyne/A)")
for ii in range(len(final_mol.frequencies)):
    print("%d   %13.4f   %13.4f   %13.4f"%(ii,final_mol.frequencies[ii],final_mol.reduced_masses[ii],final_mol.force_constants[ii]))

