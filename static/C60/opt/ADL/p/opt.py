import mlatom as ml

# dir = '/export/home/huangyh/projects/pve_2/tri/ADL/crest/lll6/min'
mol = ml.data.molecule.load('c66h10_pro_gaussian.log',format='gaussian')

xtb = ml.models.methods(method='GFN2-xTB',save_files_in_current_directory=False)
delta_model = ml.models.ani(model_file='../ADL_main_model_C60_butadiene.npz',device='cpu')
xtb_node = ml.models.model_tree_node(name='xtb',operator='predict',model=xtb)
delta_model_node = ml.models.model_tree_node(name='delta',operator='predict',model=delta_model)
method = ml.models.model_tree_node(name='model',operator='sum',children=[xtb_node,delta_model_node])

geomopt = ml.optimize_geometry(ts=False,
                               model=method,
                               initial_molecule=mol,
                               program='gaussian',
                               program_kwargs={'opt_keywords': ['calcfc', 'nomicro']}
                               )

# Get the final geometry approximately at the full CI level
final_mol = geomopt.optimized_molecule
print('Optimized coordinates:')
print(final_mol.get_xyz_string())
final_mol.write_file_with_xyz_coordinates(filename='final_p.xyz')

# Let's check how many full CI calculations, our delta-model saved us
print('Number of optimization steps:', len(geomopt.optimization_trajectory.steps))

# Calculate frequency
freq = ml.freq(model=method, molecule=final_mol, program='gaussian')

# Check vibration analysis
print("Mode     Frequencies     Reduced masses     Force Constants")
print("           (cm^-1)            (AMU)           (mDyne/A)")
for ii in range(len(final_mol.frequencies)):
    print("%d   %13.4f   %13.4f   %13.4f"%(ii,final_mol.frequencies[ii],final_mol.reduced_masses[ii],final_mol.force_constants[ii]))
