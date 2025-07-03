import mlatom as ml
import numpy as np 
import os 
from delta_al import * 


b3lyp = ml.models.methods(method='UB3LYP/6-31G*',program='Gaussian',nthreads=1,save_files_in_current_directory=False)
b3lyp_node = ml.models.model_tree_node(name='b3lyp',operator='predict',model=b3lyp)
xtb = ml.models.methods(method='GFN2-xTB*',save_files_in_current_directory=False)
xtb_node = ml.models.model_tree_node(name='xtb',operator='predict',model=xtb)
d4 = ml.models.methods(method='D4',functional='b3lyp',save_files_in_current_directory=False,nthreads=1)
d4_node = ml.models.model_tree_node(name='d4',operator='predict',model=d4)

baseline_method = ml.models.model_tree_node(children=[xtb_node,d4_node],operator='sum')

# model training setting
ref = my_reference_method(baseline=xtb,reference=b3lyp)
# MD running setting
ml_model = delta_ml_model
ml_model_kwargs = {
    'ml_model_type':'ANI',
    'baseline':baseline_method,
}
# eqmol = eqmol.copy(atomic_labels=['xyz_coordinates'])
eqmol = ml.data.molecule()
eqmol.load('c60_ts_ub3lypd4.json',format='json')

al = ml.active_learning(
    # molecule=eqmol,
    reference_method=ref,
    label_nthreads=28,
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
    min_new_points=5,
)