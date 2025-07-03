import mlatom as ml

# dir = '/export/home/huangyh/projects/pve_2/tri/ADL/crest/lll6/min'
mol = ml.data.molecule.load('c66h10_pro_gaussian.log',format='gaussian')

b3lyp = ml.models.methods(method='UB3LYP/6-31G*',program='Gaussian',nthreads=1,save_files_in_current_directory=False)
d4 = ml.models.methods(method='D4', functional='b3lyp',save_files_in_current_directory=False,nthreads=1)
b3lyp_model_tree = ml.models.model_tree_node(name='b3lyp',operator='predict',model=b3lyp)
d4_model_tree = ml.models.model_tree_node(name='d4',operator='predict',model=d4)
method = ml.models.model_tree_node(name='ub3lyp_d4',children=[b3lyp_model_tree,d4_model_tree],operator='sum')

method.predict(molecule=mol, calculate_energy=True)

print(mol.energy)