import mlatom as ml 
import numpy as np 
import os 
import matplotlib.pyplot as plt 
import scipy

class delta_ml_model(ml.al_utils.ml_model):

    def __init__(self,al_info={},model_file=None,device=None,verbose=False,ml_model_type='ANI',baseline=None,validation_set_fraction=0.1,**kwargs):
        super().__init__(
            al_info=al_info,
            model_file=model_file,
            device=device,
            verbose=verbose,
            ml_model_type=ml_model_type,
            **kwargs
        )
        self.baseline = baseline 
        self.validation_set_fraction = validation_set_fraction

    def train(self,molecular_database=None,al_info={}):
        if 'working_directory' in al_info.keys():
            workdir = al_info['working_directory']
            if self.ml_model_type.casefold() == 'kreg':
                self.main_model.model_file = os.path.join(workdir,f"{self.model_file}.npz")
                self.aux_model.model_file = os.path.join(workdir,f"aux_{self.model_file}.npz")
            else:
                self.main_model.model_file = os.path.join(workdir,f"{self.model_file}.pt")
                self.aux_model.model_file = os.path.join(workdir,f"aux_{self.model_file}.pt")
        else:
            workdir='.'
        if not os.path.exists(os.path.join(workdir,'training_db.json')):
            [subtraindb, valdb] = molecular_database.split(number_of_splits=2, fraction_of_points_in_splits=[1-self.validation_set_fraction, self.validation_set_fraction], sampling='random')
            trainingdb = subtraindb+valdb
            trainingdb.dump(os.path.join(workdir,'training_db.json'),format='json')
        else:
            trainingdb = ml.data.molecular_database.load(filename=os.path.join(workdir,'training_db.json'),format='json')
            Nsubtrain = int(len(trainingdb)*(1-self.validation_set_fraction))
            subtraindb = trainingdb[:Nsubtrain]
            valdb = trainingdb[Nsubtrain:]

        # train the main model on delta energies and gradients
        if not os.path.exists(self.main_model.model_file):
            self.main_model = self.initialize_model(ml_model_type=self.ml_model_type,
                                                    model_file=self.main_model.model_file,
                                                    device=self.device,
                                                    verbose=self.verbose)
            self.model_trainer(
                ml_model_type=self.ml_model_type,
                model=self.main_model,
                subtraindb=subtraindb,
                valdb=valdb,
                learning_grad=True,
            )
        else:
            print(f"Model file {self.main_model.model_file} found, skip training")
            self.main_model = self.initialize_model(ml_model_type=self.ml_model_type,
                                                model_file=self.main_model.model_file,
                                                device=self.device,
                                                verbose=self.verbose)
            
        # train the auxiliary model on reference energies and energy gradients
        if not os.path.exists(self.aux_model.model_file):
            self.aux_model = self.initialize_model(ml_model_type=self.ml_model_type,
                                               model_file=self.aux_model.model_file,
                                               device=self.device,
                                               verbose=self.verbose)
            self.model_trainer(
                ml_model_type=self.ml_model_type,
                model=self.aux_model,
                subtraindb=subtraindb,
                valdb=valdb,
                learning_grad=False,
            )
        else:
            print(f"Model file {self.aux_model.model_file} found, skip training")
            self.aux_model = self.initialize_model(ml_model_type=self.ml_model_type,
                                               model_file=self.aux_model.model_file,
                                               device=self.device,
                                               verbose=self.verbose)
            
        # UQ threshold 
        if not 'uq_threshold' in al_info.keys():
            # Make a copy of valdb 
            valdb_copy = valdb.copy()
            self.predict(molecular_database=valdb_copy)
            uqs = valdb_copy.get_properties('uq')
            al_info['uq_threshold'] = self.threshold_metric(uqs,metric='m+3mad')
            print(f"New threshold: {al_info['uq_threshold']}")
        else:
            print(f"Current threshold: {al_info['uq_threshold']}")
        self.uq_threshold = al_info['uq_threshold']

        # Update model file in al info 
        al_info['main_mlmodel_file'] = self.main_model.model_file 
        al_info['aux_mlmodel_file'] = self.aux_model.model_file

        self.summary(subtraindb=subtraindb,valdb=valdb,workdir=workdir)


    def predict(self,molecule=None,molecular_database=None,**kwargs):
        if not molecule is None:
            molecular_database = ml.data.molecular_database(molecule)
        else:
            if molecular_database is None:
                raise ValueError("Please provide molecule or molecular database")
            
        # Get delta energies and energy gradients from the main model 
        self.main_model.predict(
            molecular_database=molecular_database,
            property_to_predict="delta_energy",
            xyz_derivative_property_to_predict="delta_energy_gradients"
        )

        # Calculate baseline energies and energy gradients 
        moldb_copy = molecular_database.copy(atomic_labels=["xyz_coordinates"])
        self.baseline.predict(
            molecular_database=moldb_copy,
            calculate_energy=True,
            calculate_energy_gradients=True
        )

        # Add delta energies to baseline energies
        for imol,mol in enumerate(molecular_database):
            mol.energy = mol.delta_energy + moldb_copy[imol].energy 
            mol.add_xyz_vectorial_property(
                mol.get_xyz_vectorial_properties('delta_energy_gradients')+moldb_copy[imol].get_xyz_vectorial_properties('energy_gradients'),
                'energy_gradients'
            )

        # # Get energies from the aux model
        self.aux_model.predict(
            molecular_database=molecular_database,
            property_to_predict="aux_delta_energy",
            xyz_derivative_property_to_predict="aux_delta_energy_gradients"
        )

        # Calculate uncertainties 
        for mol in molecular_database:
            mol.uq = abs(mol.delta_energy-mol.aux_delta_energy)
            # print(mol.uq)
            if not self.uq_threshold is None:
                if mol.uq > self.uq_threshold:
                    mol.uncertain = True 
                else:
                    mol.uncertain = False

    def model_trainer(self,ml_model_type,model,subtraindb,valdb,learning_grad=False):
        # Make a clean copy of the databset
        subtraindb_copy = subtraindb.copy()
        valdb_copy = valdb.copy()

        # Energy is needed to get the equilibrium molecule in KREG model
        if ml_model_type.casefold() == 'kreg':
            for mol in subtraindb_copy+valdb_copy:
                mol.energy = mol.reference_energy

        # Train main model
        if learning_grad:
            if ml_model_type.casefold() == 'ani':
                model.train(
                    molecular_database=subtraindb_copy,
                    validation_molecular_database=valdb_copy,
                    property_to_learn='delta_energy',
                    xyz_derivative_property_to_learn='delta_energy_gradients'
                )
            elif ml_model_type.casefold() == 'kreg':
                model_file_saved = model.model_file
                model.model_file = 'mlmodel.npz'
                model.hyperparameters['sigma'].minval = 2**-5
                model.optimize_hyperparameters(subtraining_molecular_database=subtraindb_copy,
                                                    validation_molecular_database=valdb_copy,
                                                optimization_algorithm='grid',
                                                hyperparameters=['lambda', 'sigma'],
                                                training_kwargs={'property_to_learn': 'delta_energy', 'xyz_derivative_property_to_learn':'delta_energy_gradients', 'prior': 'mean'},
                                                prediction_kwargs={'property_to_predict': 'estimated_delta_energy','xyz_derivative_property_to_predict':'estimated_delta_energy_gradients'})
                lmbd_ = model.hyperparameters['lambda'].value ; sigma_ = model.hyperparameters['sigma'].value
                print(f"Optimized hyperparameters for main model: lambda={lmbd_}, sigma={sigma_}")
                model.model_file = model_file_saved
                model.kreg_api.save_model(model.model_file)
            elif ml_model_type.casefold() == 'mace':
                model.train(
                    molecular_database=subtraindb_copy,
                    validation_molecular_database=valdb_copy,
                    property_to_learn='delta_energy',
                    xyz_derivative_property_to_learn='delta_energy_gradients'
                )
        # Train aux model
        else:
            if ml_model_type.casefold() == 'ani':
                model.train(
                    molecular_database=subtraindb_copy,
                    validation_molecular_database=valdb_copy,
                    property_to_learn='delta_energy'
                )
            elif ml_model_type.casefold() == 'kreg':
                model_file_saved = model.model_file
                model.model_file = 'mlmodel.npz'
                model.hyperparameters['sigma'].minval = 2**-5
                model.optimize_hyperparameters(subtraining_molecular_database=subtraindb_copy,
                                                    validation_molecular_database=valdb_copy,
                                                    optimization_algorithm='grid',
                                                    hyperparameters=['lambda','sigma'],
                                                    training_kwargs={'property_to_learn': 'delta_energy', 'prior': 'mean'},
                                                    prediction_kwargs={'property_to_predict': 'estimated_energy',})
                lmbd_ = model.hyperparameters['lambda'].value ; sigma_ = model.hyperparameters['sigma'].value
                # self.hyperparameters['lambda'] = lmbd_; self.hyperparameters['sigma'] = sigma_
                print(f"Optimized hyperparameters for aux model: lambda={lmbd_}, sigma={sigma_}")
                model.model_file = model_file_saved
                model.kreg_api.save_model(model.model_file)
            elif ml_model_type.casefold() == 'mace':
                model.train(
                    molecular_database=subtraindb_copy,
                    validation_molecular_database=valdb_copy,
                    property_to_learn='delta_energy'
                )

    def summary(self,subtraindb,valdb,workdir):
        Nsubtrain = len(subtraindb)
        Nvalidate = len(valdb)
        Ntrain = Nsubtrain + Nvalidate
        trainingdb_ref = subtraindb + valdb 
        trainingdb = trainingdb_ref.copy()
        self.predict(molecular_database=trainingdb)
        print(f'    Number of training points: {Nsubtrain+Nvalidate}')
        print(f'        Number of subtraining points: {Nsubtrain}')
        print(f'        Number of validation points: {Nvalidate}')
        values = trainingdb_ref.get_properties('delta_energy')
        estimated_values = trainingdb.get_properties('delta_energy')
        aux_estimated_values = trainingdb.get_properties('aux_delta_energy')

        gradients = trainingdb_ref.get_xyz_vectorial_properties('delta_energy_gradients')
        estimated_gradients = trainingdb.get_xyz_vectorial_properties('delta_energy_gradients')
        # aux_estimated_gradients = trainingdb.get_xyz_vectorial_properties('aux_energy_gradients') - trainingdb_ref.get_xyz_vectorial_properties('baseline_energy_gradients')

        # Evaluate main model performance 
        # .RMSE of values
        main_model_subtrain_vRMSE = ml.stats.rmse(estimated_values[:Nsubtrain],values[:Nsubtrain])
        main_model_validate_vRMSE = ml.stats.rmse(estimated_values[Nsubtrain:],values[Nsubtrain:])
        # .Pearson correlation coefficient of values
        main_model_subtrain_vPCC = ml.stats.correlation_coefficient(estimated_values[:Nsubtrain],values[:Nsubtrain])
        main_model_validate_vPCC = ml.stats.correlation_coefficient(estimated_values[Nsubtrain:],values[Nsubtrain:])
        # .RMSE of gradients
        try:
            main_model_subtrain_gRMSE = ml.stats.rmse(estimated_gradients[:Nsubtrain].flatten(),gradients[:Nsubtrain].flatten())
            main_model_validate_gRMSE = ml.stats.rmse(estimated_gradients[Nsubtrain:].flatten(),gradients[Nsubtrain:].flatten())
        except:
            main_model_subtrain_gRMSE = 'not calculated'
            main_model_validate_gRMSE = 'not calculated'
        # .Pearson correlation coeffcient of gradients 
        try:
            main_model_subtrain_gPCC = ml.stats.correlation_coefficient(estimated_gradients[:Nsubtrain].flatten(),gradients[:Nsubtrain].flatten())
            main_model_validate_gPCC = ml.stats.correlation_coefficient(estimated_gradients[Nsubtrain:].flatten(),gradients[Nsubtrain:].flatten())
        except:
            main_model_subtrain_gPCC = 'not calculated'
            main_model_validate_gPCC = 'not calculated'
        # Evaluate auxiliary model performance
        # .RMSE of values 
        aux_model_subtrain_vRMSE = ml.stats.rmse(aux_estimated_values[:Nsubtrain],values[:Nsubtrain])
        aux_model_validate_vRMSE = ml.stats.rmse(aux_estimated_values[Nsubtrain:],values[Nsubtrain:])
        # .Pearson correlation coefficient of values
        aux_model_subtrain_vPCC = ml.stats.correlation_coefficient(aux_estimated_values[:Nsubtrain],values[:Nsubtrain])
        aux_model_validate_vPCC = ml.stats.correlation_coefficient(aux_estimated_values[Nsubtrain:],values[Nsubtrain:])
        # .RMSE of gradients
        # try:
        #     aux_model_subtrain_gRMSE = ml.stats.rmse(aux_estimated_gradients[:Nsubtrain].flatten(),gradients[:Nsubtrain].flatten())
        #     aux_model_validate_gRMSE = ml.stats.rmse(aux_estimated_gradients[Nsubtrain:].flatten(),gradients[Nsubtrain:].flatten())
        # except:
        #     aux_model_subtrain_gRMSE = 'not calculated'
        #     aux_model_validate_gRMSE = 'not calculated'
        # # .Pearson correlation coeffcient of gradients 
        # try:
        #     aux_model_subtrain_gPCC = ml.stats.correlation_coefficient(aux_estimated_gradients[:Nsubtrain].flatten(),gradients[:Nsubtrain].flatten())
        #     aux_model_validate_gPCC = ml.stats.correlation_coefficient(aux_estimated_gradients[Nsubtrain:].flatten(),gradients[Nsubtrain:].flatten())
        # except:
        #     aux_model_subtrain_gPCC = 'not calculated'
        #     aux_model_validate_gPCC = 'not calculated'

        print("        Main model")
        print("            Subtraining set:")
        print(f"                RMSE of values = {main_model_subtrain_vRMSE}")
        print(f"                Correlation coefficient = {main_model_subtrain_vPCC}")
        print(f"                RMSE of gradients = {main_model_subtrain_gRMSE}")
        print(f"                Correlation coefficient = {main_model_subtrain_gPCC}")
        print("            Validation set:")
        print(f"                RMSE of values = {main_model_validate_vRMSE}")
        print(f"                Correlation coefficient = {main_model_validate_vPCC}")
        print(f"                RMSE of gradients = {main_model_validate_gRMSE}")
        print(f"                Correlation coefficient = {main_model_validate_gPCC}")
        print("        Auxiliary model")
        print("            Subtraining set:")
        print(f"                RMSE of values = {aux_model_subtrain_vRMSE}")
        print(f"                Correlation coefficient = {aux_model_subtrain_vPCC}")
        # print(f"                RMSE of gradients = {aux_model_subtrain_gRMSE}")
        # print(f"                Correlation coefficient = {aux_model_subtrain_gPCC}")
        print("            Validation set:")
        print(f"                RMSE of values = {aux_model_validate_vRMSE}")
        print(f"                Correlation coefficient = {aux_model_validate_vPCC}")
        # print(f"                RMSE of gradients = {aux_model_validate_gRMSE}")
        # print(f"                Correlation coefficient = {aux_model_validate_gPCC}")

        # Value scatter plot of the main model
        fig,ax = plt.subplots() 
        fig.set_size_inches(15,12)
        diagonal_line = [min([min(values),min(estimated_values)]),max([max(values),max(estimated_values)])]
        ax.plot(diagonal_line,diagonal_line,color='C3')
        ax.scatter(values[0:Nsubtrain],estimated_values[0:Nsubtrain],color='C0',label='subtraining points')
        ax.scatter(values[Nsubtrain:Ntrain],estimated_values[Nsubtrain:Ntrain],color='C1',label='validation points')
        ax.set_xlabel(f'Energy (Hartree)')
        ax.set_ylabel(f'Estimated energy (Hartree)')
        plt.suptitle(f'Main model (energies)')
        plt.legend()
        plt.savefig(os.path.join(workdir,'mlmodel_energies.png'),dpi=300)
        fig.clear()
        # Gradient scatter plot of the main model 
        try:
            fig,ax = plt.subplots()
            fig.set_size_inches(15,12)
            diagonal_line = [min([np.min(gradients),np.min(estimated_gradients)]),max([np.max(gradients),np.max(estimated_gradients)])]
            ax.plot(diagonal_line,diagonal_line,color='C3')
            ax.scatter(gradients[0:Nsubtrain].flatten(),estimated_gradients[0:Nsubtrain].flatten(),color='C0',label='subtraining points')
            ax.scatter(gradients[Nsubtrain:Ntrain].flatten(),estimated_gradients[Nsubtrain:Ntrain].flatten(),color='C1',label='validation points')
            ax.set_xlabel(f'Energy gradients (Hartree/Angstrom)')
            ax.set_ylabel(f'Estimated energy gradients (Hartree/Angstrom)')
            ax.set_title(f'Main model (energy gradients)')
            plt.legend()
            plt.savefig(os.path.join(workdir,'mlmodel_energy_gradients.png'),dpi=300)
            fig.clear()
        except:
            print('Cannot plot gradients plot of main model')

        # Value scatter plot of the auxiliary model
        fig,ax = plt.subplots() 
        fig.set_size_inches(15,12)
        diagonal_line = [min([min(values),min(aux_estimated_values)]),max([max(values),max(aux_estimated_values)])]
        ax.plot(diagonal_line,diagonal_line,color='C3')
        ax.scatter(values[0:Nsubtrain],aux_estimated_values[0:Nsubtrain],color='C0',label='subtraining points')
        ax.scatter(values[Nsubtrain:Ntrain],aux_estimated_values[Nsubtrain:Ntrain],color='C1',label='validation points')
        ax.set_xlabel(f'Energy (Hartree)')
        ax.set_ylabel(f'Estimated energy (Hartree)')
        ax.set_title(f'Auxiliary model (energies)')
        plt.legend()
        plt.savefig(os.path.join(workdir,'aux_mlmodel_energies.png'),dpi=300)


class my_reference_method(ml.models.model):
    def __init__(self,baseline,reference):
        self.baseline = baseline 
        self.reference = reference 

    def predict(self,molecule=None,molecular_database=None,**kwargs):
        if not molecule is None:
            molecular_database = ml.data.molecular_database(molecule)
        else:
            if molecular_database is None:
                raise ValueError("Please provide molecule or molecular database")
        # Check if the molecules are already labeled 
        moldb2label = ml.data.molecular_database()
        for mol in molecular_database:
            # print(mol.__dict__)
            if not 'delta_energy' in mol.__dict__:
                moldb2label += mol
        # Make a clean copy of molecular database 
        # try:
        moldb_baseline = moldb2label.copy()
        self.baseline.predict(molecular_database=moldb_baseline,calculate_energy=True,calculate_energy_gradients=True,calculate_hessian=False)
        moldb_reference = moldb2label.copy()
        for mol in moldb_reference:
            print(mol.id)
        self.reference.predict(molecular_database=moldb_reference,calculate_energy=True,calculate_energy_gradients=True,calculate_hessian=False)
        # except:
        #     print("An error occurred in predicting, results will not be saved in the molecule")
        #     moldb_reference = moldb2label.copy()
        #     moldb_baseline = moldb2label.copy()
        # Save properties in the input molecular database
        for imol, mol in enumerate(moldb2label):
            if 'energy' in moldb_reference[imol].__dict__ and 'energy' in moldb_baseline[imol].__dict__:
                mol.reference_energy = moldb_reference[imol].energy 
                mol.add_xyz_vectorial_property(moldb_reference[imol].get_xyz_vectorial_properties('energy_gradients'),'reference_energy_gradients')
                mol.delta_energy = moldb_reference[imol].energy - moldb_baseline[imol].energy 
                mol.baseline_energy = moldb_baseline[imol].energy 
                mol.add_xyz_vectorial_property(moldb_baseline[imol].get_xyz_vectorial_properties('energy_gradients'),'baseline_energy_gradients')
                mol.add_xyz_vectorial_property(moldb_reference[imol].get_xyz_vectorial_properties('energy_gradients')-moldb_baseline[imol].get_xyz_vectorial_properties('energy_gradients'),'delta_energy_gradients')
                mol.failed = False
            else:
                mol.failed = True

    @property 
    def nthreads(self):
        return self._nthreads 
    
    @nthreads.setter 
    def nthreads(self,value):
        self._nthreads = value
        self.baseline.nthreads = value 
        self.reference.nthreads = value 
