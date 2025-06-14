import mlatom as ml

# dir = '/export/home/huangyh/projects/pve_2/tri/ADL/crest/lll6/min'
mol = ml.data.molecule.load('b3_p.xyz',format='xyz')

method = ml.models.ani(model_file='../AL_main_model_ethene_butadiene.npz',device='cpu')

method.predict(molecule=mol, calculate_energy=True)

print(mol.energy)
