import mlatom as ml

# dir = '/export/home/huangyh/projects/pve_2/tri/ADL/crest/lll6/min'
mol = ml.data.molecule.load('b3_r.xyz',format='xyz')

xtb = ml.models.methods(method='GFN2-xTB',save_files_in_current_directory=False)
delta_model = ml.models.ani(model_file='../ADL_main_model_ethene_butadiene.npz',device='cpu')
xtb_node = ml.models.model_tree_node(name='xtb',operator='predict',model=xtb)
delta_model_node = ml.models.model_tree_node(name='delta',operator='predict',model=delta_model)
method = ml.models.model_tree_node(name='model',operator='sum',children=[xtb_node,delta_model_node])

method.predict(molecule=mol, calculate_energy=True)

print(mol.energy)
