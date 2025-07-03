import mlatom as ml

# dir = '/export/home/huangyh/projects/pve_2/tri/ADL/crest/lll6/min'
mol = ml.data.molecule.load('b3_r.xyz',format='xyz')

method = ml.models.methods(method='B3LYP/6-31G*',program='Gaussian',save_files_in_current_directory=False)

method.predict(molecule=mol, calculate_energy=True)

print(mol.energy)

