import mlatom as ml
import numpy as np 
import os 
from delta_al import * 

# Baseline method 
# xtb = ml.models.methods(method='GFN2-xTB*',nthreads=1)
ani1ccx = ml.models.methods(method='ANI-1ccx')
# Reference method
# b3lyp = ml.models.methods(method='B3LYP/6-31G*',program='Gaussian',nthreads=1)
b3lyp = ml.models.methods(method='UB3LYP/6-31G*',program='Gaussian',nthreads=20,save_files_in_current_directory=False)
d4 = ml.models.methods(method='D4', functional='b3lyp',save_files_in_current_directory=False,nthreads=20)
b3lyp_model_tree = ml.models.model_tree_node(name='b3lyp',operator='predict',model=b3lyp)
d4_model_tree = ml.models.model_tree_node(name='d4',operator='predict',model=d4)
method = ml.models.model_tree_node(name='ub3lyp_d4',children=[b3lyp_model_tree,d4_model_tree],operator='sum')
# Optimize geometry and calculate frequencies
# if not os.path.exists('ethanol_eqmol.json'):
#     molecule = ml.data.molecule.from_xyz_file('ethanol.xyz')
#     eqmol = ml.optimize_geometry(model=xtb,initial_molecule=molecule).optimized_molecule
#     freq = ml.freq(model=xtb,molecule=eqmol)
#     eqmol.dump('ethanol_eqmol.json',format='json')
# else:
#     eqmol = ml.data.molecule() 
#     eqmol.load('ethanol_eqmol.json',format='json')

ref = my_reference_method(baseline=ani1ccx,reference=method)
ml_model = delta_ml_model
ml_model_kwargs = {
    'ml_model_type':'ANI',
    'baseline':ani1ccx,
}
# eqmol = eqmol.copy(atomic_labels=['xyz_coordinates'])
eqmol = ml.data.molecule()
eqmol.load('ts.json',format='json')

al = ml.al(
    # molecule=eqmol,
    reference_method=ref,
    label_nthreads=1,
    initdata_sampler='harmonic-quantum-boltzmann',
    initdata_sampler_kwargs={
        'molecule':eqmol,
        'number_of_initial_conditions':250,
        'initial_temperature':298,
    },
    initial_points_refinement='one-shot',
    sampler='batch_md',
    sampler_kwargs={
        'initcond_sampler':'harmonic-quantum-boltzmann',
        'initcond_sampler_kwargs':{
            'molecule':eqmol,
            'number_of_initial_conditions':100,
            'initial_temperature':298,
        },
        'maximum_propagation_time':500,
        'time_step':0.5,
        'ensemble':'NVE'
    },
    ml_model=ml_model,
    ml_model_kwargs=ml_model_kwargs,
    new_points=None,
    min_new_points=1,
)