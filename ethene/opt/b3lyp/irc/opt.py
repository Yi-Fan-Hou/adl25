import mlatom as ml

# dir = '/export/home/huangyh/projects/pve_2/tri/ADL/crest/lll6/min'
mol = ml.data.molecule.load('final_ts.xyz',format='xyz')

b3lyp = ml.models.methods(method='B3LYP/6-31G*',program='Gaussian',nthreads=6,save_files_in_current_directory=False)

IRC = ml.simulations.irc(model=b3lyp,ts_molecule=mol)


