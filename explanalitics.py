import pandas as pd
import numpy as np
import os
from itertools import product
import random

# Progress bar begin
import sys
import time
# Progress bar end

from lamp import Lamp
import altair as alt
import math

from sklearn.preprocessing import MinMaxScaler

import torch
torch.manual_seed(42)
import torch.nn as nn
from torch.autograd import Variable

from captum.attr import (
    IntegratedGradients,
    Saliency,
    DeepLift,
    InputXGradient,
    GuidedBackprop,
    FeatureAblation,
    FeaturePermutation,
    ShapleyValueSampling,
    Lime,
    KernelShap,
    Occlusion,
    NoiseTunnel,
    GradientShap,
    DeepLiftShap
)

from captum.metrics import (
    infidelity,
    sensitivity_max,
    infidelity_perturb_func_decorator,
)

class Explainers():
    
    #integrated gradients
    def lem_ig(self, model, X, params_dict=None):
        attr_method = IntegratedGradients(model)
        
        if X is None:
            return attr_method
            
        X.requires_grad_()
        if params_dict is None:
            attributions = attr_method.attribute(X)
        else:
            attributions = attr_method.attribute(X, **params_dict) 
        return attributions, attr_method

    #input x gradient
    def lem_ixg(self, model, X, params_dict=None):
        attr_method = InputXGradient(model)
        
        if X is None:
            return attr_method
            
        X.requires_grad_()
        if params_dict is None:
            attributions = attr_method.attribute(X)
        else:
            attributions = attr_method.attribute(X,**params_dict)
          
        return attributions, attr_method

    #smoothgrad
    def lem_sg(self, model, X, params_dict=None):
        attr_method = NoiseTunnel(Saliency(model))
        
        if X is None:
            return attr_method
            
        X.requires_grad_()
        if params_dict is None:
            attributions = attr_method.attribute(X, nt_type='smoothgrad', abs=False, stdevs=0.1)
        else:
            attributions = attr_method.attribute(X,**params_dict)
            
        return attributions, attr_method


    #vanilla gradient
    def lem_vg(self, model, X, params_dict=None):
        attr_method = Saliency(model)
        
        if X is None:
            return attr_method
            
        X.requires_grad_()
        if params_dict is None:
            attributions = attr_method.attribute(X, abs=False)
        else:
            attributions = attr_method.attribute(X,**params_dict)
               
        return attributions, attr_method

    #guided backpropagation
    def lem_gb(self, model, X, params_dict=None):
        attr_method = GuidedBackprop(model)
        
        if X is None:
            return attr_method
            
        X.requires_grad_()
        if params_dict is None:
            attributions = attr_method.attribute(X)
        else:
            attributions = attr_method.attribute(X,**params_dict) 
        return attributions, attr_method


    #occlusion
    def lem_oc(self, model, X, params_dict=None):
        attr_method = Occlusion(model)
        
        if X is None:
            return attr_method
            
        if params_dict is None:
            attributions = attr_method.attribute(X, sliding_window_shapes=(1,))
        else:
            attributions = attr_method.attribute(X,**params_dict)
   
        return attributions, attr_method


    #lime
    def lem_lime(self, model, X, params_dict=None):
        attr_method = Lime(model)
        
        if X is None:
            return attr_method
            
        if params_dict is None:
            attributions = attr_method.attribute(X, n_samples=100)
        else:
            attributions = attr_method.attribute(X,**params_dict)

        return attributions, attr_method


    #kernelshap
    def lem_ks(self, model, X, params_dict=None):
        attr_method = KernelShap(model)
        
        if X is None:
            return attr_method
            
        if params_dict is None:
            attributions = attr_method.attribute(X, n_samples=100)
        else:
            attributions = attr_method.attribute(X,**params_dict)

        return attributions, attr_method

    #deeplift
    def lem_dpl(self, model, X, params_dict=None):
        attr_method = KernelShap(model)
        
        if X is None:
            return attr_method
            
        if params_dict is None:
            attributions = attr_method.attribute(X)
        else:
            attributions = attr_method.attribute(X,**params_dict)
 
        return attributions, attr_method

    #gradiendshap
    def lem_gs(self, model, X, params_dict=None):
        attr_method = GradientShap(model)
        
        if X is None:
            return attr_method
            
        if params_dict is None:
            baseline_dist = torch.randn(1, X.shape[1]) * 0.001
            attributions = attr_method.attribute(X, stdevs=0.09, n_samples=100, baselines=baseline_dist)
        else:
            attributions = attr_method.attribute(X,**params_dict)
    
        return attributions, attr_method
    
    
class Projection():
    
    def apply_lamp(self, df):
        df = df.drop(['best'], axis=1).copy()
        pivoted_df = df.pivot(index='method', columns='metric', values='value')
#         print(pivoted_df)
        matrix_points = pivoted_df.to_numpy()
#         print(matrix_points)
        
        #Cria dataframe com a matrix de pontos
        df_ds = pd.DataFrame(matrix_points)
        n_ds = df_ds.shape[0]
        
        #Cria os pontos de controle
        sample_size = matrix_points.shape[1] #numero de combinacoes
        table = list(product([0, 1], repeat=sample_size))
        
        ctp_samples = []
        for i in table:
            ctp_samples.append(list(i))
            
        control_points_samples = np.array(ctp_samples)
        
        df_ctp_4d = pd.DataFrame(control_points_samples)
        n_4d = df_ctp_4d.shape[0]
        
        cp_positions = self.__control_points_position(control_points_samples, max_sum=4)
        df_ctp_2d = pd.DataFrame(cp_positions)
        
        df_ds_cct = pd.concat([df_ds, df_ctp_4d], ignore_index=True)
        
        ids = np.arange(n_ds, n_ds+n_4d)
        df_ctp_2d[2] = ids
        
        ctp_2d = df_ctp_2d.values
        data = df_ds_cct.values
        
        lamp_proj = Lamp(Xdata= data, control_points= ctp_2d, label=False)
        data_proj = lamp_proj.fit()
        
#         plt.scatter(data_proj[:,0], data_proj[:, 1], color='gray')
#         plt.scatter(ctp_2d[:,0], ctp_2d[:, 1], color='red', s=2)
        
        size_dataset = len(matrix_points)
        df_lamp = pd.DataFrame(data_proj[:size_dataset,:], columns=['Lamp_1','Lamp_2'])
        
        return df_lamp, pivoted_df.index.to_numpy()
        
    # TODO funciona para 4 metricas, precisa generalizar e com max=6
    def __control_points_position(self, samples, max_sum=4):
        positions = []

        line1 = np.arange(0.0, 0.51, (1/2)/3)
        line2 = np.arange(0.0, 1.01, 1/5)
        line3 = np.arange(0.0, 0.51, (1/2)/3)

        for point in samples:
            if np.sum(point)==0:
                positions.append(np.array([0,0]))
            elif np.sum(point)==max_sum:
                positions.append(np.array([1,1]))
            elif np.sum(point)==1:
                x=line1[0]
                y=(1/2)-line1[0]
                positions.append(np.array([x,y]))
                line1 = np.delete(line1, 0)
            elif np.sum(point)==2:
                x=line2[0]
                y=1-line2[0]
                positions.append(np.array([x,y]))
                line2 = np.delete(line2, 0)
            elif np.sum(point)==3:
                temp_x = line3+(1/2)
                x = temp_x[0]
                temp_y = 1-line3
                y = temp_y[0]
                positions.append(np.array([x,y]))
                line3 = np.delete(line3, 0)
            else:
                positions.append(np.array([-1,-1]))
                
        return positions
    
    
############################################# class from OpenXAI
class BasePerturbation:
    '''
    Base Class for perturbation methods.
    '''
    
    def __init__(self, data_format):
        '''
        Initialize generic parameters for the perturbation method
        '''
        self.data_format = data_format
    
    def get_perturbed_inputs(self):
        '''
        This function implements the logic of the perturbation methods which will return perturbed samples.
        '''
        pass
    
class NormalPerturbation(BasePerturbation):
    def __init__(self, data_format, mean: int = 0, std_dev: float = 0.05, flip_percentage: float = 0.3):
        self.mean = mean
        self.std_dev = std_dev
        self.flip_percentage = flip_percentage
        
        super(NormalPerturbation, self).__init__(data_format)
        '''
        Initializes the marginal perturbation method where each column is sampled from marginal distributions given per variable.
        dist_per_feature : vector of distribution generators (tdist under torch.distributions).
        Note : These distributions are assumed to have zero mean since they get added to the original sample.
        '''
        pass
    
    def get_perturbed_inputs(self, original_sample: torch.FloatTensor, feature_mask: torch.BoolTensor,
                             num_samples: int, feature_metadata: list) -> torch.tensor:
        '''
        feature mask : this indicates the static features
        num_samples : number of perturbed samples.
        '''
        feature_type = feature_metadata
        assert len(feature_mask) == len(original_sample),\
            f"mask size == original sample in get_perturbed_inputs for {self.__class__}"
        
        continuous_features = torch.tensor([i == 'c' for i in feature_type])
        discrete_features = torch.tensor([i == 'd' for i in feature_type])
        
        # Processing continuous columns
        perturbations = torch.normal(self.mean, self.std_dev,
                                     [num_samples, len(feature_type)]) * continuous_features + original_sample
        
        # Processing discrete columns
        flip_percentage = self.flip_percentage
        p = torch.empty(num_samples, len(feature_type)).fill_(flip_percentage)
        perturbations = perturbations * (~discrete_features) + torch.abs(
            (perturbations * discrete_features) - (torch.bernoulli(p) * discrete_features))
        
        # keeping features static that are in top-K based on feature mask
        perturbed_samples = original_sample * feature_mask + perturbations * (~feature_mask)
        
        return perturbed_samples
    
############################################# class from OpenXAI    


class Explanalytics():
    
    def __init__(self):
        self.model = None
        self.X_sample = None
        self.y_sample = None
        self.feature_names = None
        self.methods_dict_param = None
        self.metrics_used = []
        self.binarySizeOutputModel = 1
        self.directory_source = "source"
        self.directory_explanations = "feature_importance"
        self.directory_metrics = "metrics"
        self.explainers = Explainers()
        self.projection = Projection()

    def load_model(self, model, X_sample: pd.DataFrame, y_sample: pd.DataFrame):
        self.__create_directory()
        self.model = model
        
        try:
            X_sample.to_csv('./'+ self.directory_source +'/X_sample.csv', index=False)
            y_sample.to_csv('./'+ self.directory_source +'/y_sample.csv', index=False)
            
            self.X_sample = X_sample
            self.feature_names = X_sample.columns.to_list()
            self.y_sample = y_sample
        except AttributeError:
            raise AttributeError(
                f"X_sample and y_sample must be a Pandas Dataframe."
            )
            
#         with torch.no_grad():
#             y_pred = model.predict(X_sample)     
#         print(y_pred)
#         print(y_pred.shape)
#         if y_pred.shape[1] > 2:
#             raise NotImplementedError('NotImplementedError.')
            
        # transform in tensors
        x_sample_scaled_tensors = torch.tensor(X_sample.values, dtype=torch.float32)

        # we set the model in inference mode and create the predictions
        with torch.inference_mode():
            out_probs = model(x_sample_scaled_tensors)
            if out_probs.shape[1] == 1:
                self.binarySizeOutputModel = 1
                y_pred = torch.where(out_probs > 0.5, 1, 0)
            elif out_probs.shape[1] == 2:
                self.binarySizeOutputModel = 2
                y_pred = np.argmax(out_probs, axis=1)
            else:
                raise NotImplementedError('NotImplementedError.')
            
            df = pd.DataFrame()
            df['label'] = y_sample.astype(int)
            df['pred'] = y_pred
            df.to_csv('./'+ self.directory_source + '/y_pred.csv', index=False)
            
    def generate_explanations(self, methods_dict):
        self.__create_directory_explanations()
        
        self.methods_dict_param = methods_dict
        
        explainer_attr_dict = {
            'integrated gradients' : self.explainers.lem_ig, 
            'input x gradient' : self.explainers.lem_ixg, 
            'smoothgrad' : self.explainers.lem_sg,
            'vanilla gradient' : self.explainers.lem_vg, 
            'guided backprop' : self.explainers.lem_gb,
            'occlusion' : self.explainers.lem_oc,
            'lime' : self.explainers.lem_lime,
            'kernelshap' : self.explainers.lem_ks,
            'deeplift': self.explainers.lem_dpl,
            'gradientshap': self.explainers.lem_gs
        }
        
        for method in methods_dict:
            attributions_feature_importance = []
            func_explanation_method = explainer_attr_dict[method]
            params = methods_dict[method]
            for input_ in self.X_sample.to_numpy():
                input_ = torch.from_numpy(input_.astype(np.float32))
                input_ = Variable(torch.FloatTensor(input_), requires_grad=True)
                input_ = torch.reshape(input_, (1, self.X_sample.shape[1]))
                
                attributions, _ = func_explanation_method(self.model, input_, params)
                attributions = attributions.detach().numpy()
                attributions_feature_importance.append(attributions[0])
            feature_importance = pd.DataFrame(attributions_feature_importance, columns=self.feature_names)
            feature_importance.to_csv('./'+ self.directory_source + '/' + self.directory_explanations + '/'+ method +'.csv', index=False)
            
            
    def run_metric_infidelity(self, n_repetitions=1):
        
        metric_name = 'infidelity'
        self.__create_directory_metrics(metric_name)
        self.metrics_used.append(metric_name)
    
        def perturb_fn(inputs):
            noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float()
            return noise, inputs - noise

        for method in self.methods_dict_param:
            
            scores = {}
            for n in range(0, n_repetitions):
                scores[n] = []
                
            attributions_file_path = self.directory_source + '/' +self.directory_explanations + '/' + method + '.csv'
            df = pd.read_csv(attributions_file_path)
            
            for index, row in df.iterrows():
                attribution = torch.from_numpy(row.to_numpy())
                attribution = torch.reshape(attribution, (1, len(df.columns)))

                input_ = self.X_sample.iloc[[index]].to_numpy()
                input_ = torch.from_numpy(input_.astype(np.float32))
                input_ = Variable(torch.FloatTensor(input_), requires_grad=True)
                
                for n in range(0, n_repetitions):                    
                    infid = infidelity(self.model, perturb_fn, input_, attribution)
                    inf_value = infid.detach().numpy()[0]
                    scores[n].append(inf_value)
            df_scores = pd.DataFrame(scores)
            
            df_scores['average'] = df_scores.iloc[:, :n_repetitions].mean(axis=1)
            df_scores.to_csv('./'+ self.directory_source + '/' + self.directory_metrics + '/' + metric_name +
                             '/'+ method +'.csv', index=False)
            
            
    def run_metric_sensitivity(self, n_repetitions=1):
        
        metric_name = 'sensitivity'
        self.__create_directory_metrics(metric_name)
        self.metrics_used.append(metric_name)
        
        explainer_attr_dict = {
            'integrated gradients' : self.explainers.lem_ig, 
            'input x gradient' : self.explainers.lem_ixg, 
            'smoothgrad' : self.explainers.lem_sg,
            'vanilla gradient' : self.explainers.lem_vg, 
            'guided backprop' : self.explainers.lem_gb,
            'occlusion' : self.explainers.lem_oc,
            'lime' : self.explainers.lem_lime,
            'kernelshap' : self.explainers.lem_ks,
            'deeplift': self.explainers.lem_dpl,
            'gradientshap': self.explainers.lem_gs
        }
        

        for method in self.methods_dict_param:
            
            scores = {}
            for n in range(0, n_repetitions):
                scores[n] = []
            
            func_explanation_method = explainer_attr_dict[method]
            params = self.methods_dict_param[method]

            for index, row in self.X_sample.iterrows():
                input_ = self.X_sample.iloc[[index]].to_numpy()
                input_ = torch.from_numpy(input_.astype(np.float32))
                input_ = Variable(torch.FloatTensor(input_), requires_grad=True)
                
                attr_method = func_explanation_method(self.model, None, None)
                
                for n in range(0, n_repetitions):
                    if params is None:
                        sens = sensitivity_max(attr_method.attribute, input_)
                    else:
                        sens = sensitivity_max(attr_method.attribute, input_, **params)
                    sens_value = sens.detach().numpy()[0]
                    scores[n].append(sens_value)
                    
            df_scores = pd.DataFrame(scores)
            df_scores['average'] = df_scores.iloc[:, :n_repetitions].mean(axis=1)
            df_scores.to_csv('./'+ self.directory_source + '/' + self.directory_metrics + '/' + metric_name +
                             '/'+ method +'.csv', index=False)  
            
            
    def run_metric_res(self, p_norm=2, n_repetitions=3):
        metric_name = 'res'
        self.__create_directory_metrics(metric_name)
        self.metrics_used.append(metric_name)
        
        model = self.model
        data = self.X_sample
        labels = self.y_sample
        
        for method in self.methods_dict_param:
            print(method)
            stability_ratios = []
            runs= int(n_repetitions)
            data_size= data.shape[0]

            for i_data in self.__progressbar(range(data_size), "Progress: ", 40):

                # i_data and its label as pd.DataFrames
                target_x= pd.DataFrame(data=[data.iloc[i_data,:]], columns=data.columns)
                target_y= pd.DataFrame(data=[labels.iloc[i_data]], columns=labels.columns)

                # i_data and its label as tensors
                x_data = torch.tensor(target_x.values, dtype=torch.float32)
                y_data= torch.tensor(np.asarray(target_y), dtype=torch.float64)


                # data point prediction
                fx_data= model(x_data)
                y_pred = torch.where(fx_data > 0.5, 1, 0)


                x_exps= []


                for i in range(runs):
                    # ------------------------------------ get x_data explanation -- n runs

                    x_exp= pd.read_csv(self.directory_source + '/' + self.directory_explanations + '/' + method + '.csv')


                    # get the explanation of each method to each run
                    x_exps.append(x_exp.to_numpy())


                x_exps= np.asarray(x_exps)
                x_exps_mean= np.mean(x_exps, axis=0)
                x_exp_ratios= []


                # -------- distance of each explanation from the mean of explanations 
                for j in range(runs):
                    x_exp_ratios.append(self.__lp_norm_dif(x_exps_mean, x_exps[j], 
                                              p_norm=p_norm, norm=False))


                # --------------- max ratio related to each x_data
                stability_ratios.append(x_exp_ratios[np.argmax(x_exp_ratios)])

            results = pd.DataFrame()
            stability_ratios= pd.DataFrame(data= np.asarray(stability_ratios).T, 
                                                columns=['max'])
            results = pd.concat([results, stability_ratios], axis=1)
            results.to_csv('./'+ self.directory_source + '/' + self.directory_metrics + '/' + metric_name +
                                 '/'+ method +'.csv', index=False)
        
            
#     def run_metric_pgi(self, n_repetitions=1):
#         metric_name = 'pgi'
#         self.__create_directory_metrics(metric_name)
#         self.metrics_used.append(metric_name)
        
#         for method in self.methods_dict_param:
#             df_scores = pd.DataFrame()
#             df_scores['average'] = np.absolute(np.random.normal(0, 1, self.X_sample.shape[0]))
#             df_scores.to_csv('./'+ self.directory_source + '/' + self.directory_metrics + '/' + metric_name +
#                                  '/'+ method +'.csv', index=False) 
        
#     def run_metric_ris(self, n_repetitions=1):
#         metric_name = 'ris'
#         self.__create_directory_metrics(metric_name)
#         self.metrics_used.append(metric_name)
        
#         for method in self.methods_dict_param:
#             df_scores = pd.DataFrame()
#             df_scores['average'] = np.random.randint(0,150,size=self.X_sample.shape[0])
#             df_scores.to_csv('./'+ self.directory_source + '/' + self.directory_metrics + '/' + metric_name +
#                                  '/'+ method +'.csv', index=False)
            
    def run_metric_ris_ros(self, perturbation=None, descriptor=None, stab_type='both'):
        
        model = self.model
        data = self.X_sample
        labels = self.y_sample
        
        if not ('ros' in stab_type):
            metric_name = 'ris'
            self.__create_directory_metrics(metric_name)
            self.metrics_used.append(metric_name)

        if not ('ris' in stab_type):
            metric_name = 'ros'
            self.__create_directory_metrics(metric_name)
            self.metrics_used.append(metric_name)
        
        
        if perturbation is None:
            perturbation = self.__get_pertubation_func()
        if descriptor is None:
            descriptor = self.__get_descriptor()

        for method in self.methods_dict_param:
            print(method)
            ris_max_ratios= []
            ris_mean_ratios= []
            ros_max_ratios= []
            ros_mean_ratios= []
            data_size= data.shape[0]

            for i_data in self.__progressbar(range(data_size), "Progress: ", 40):

                # i_data and its label as pd.DataFrames
                target_x= pd.DataFrame(data=[data.iloc[i_data,:]], columns=data.columns)
                target_y= pd.DataFrame(data=[labels.iloc[i_data]], columns=labels.columns)

                # i_data and its label as tensors
                x_data = torch.tensor(target_x.values, dtype=torch.float32)
                y_data= torch.tensor(np.asarray(target_y), dtype=torch.float64)

                # data point prediction
                fx_data= model(x_data)
                y_pred = torch.where(fx_data > 0.5, 1, 0)

                # ------------- get x_data explanation
                x_exps = pd.read_csv(self.directory_source + '/' + self.directory_explanations + '/' + method + '.csv')
                x_exp = torch.from_numpy(x_exps.values)        

                # -------- x_data perturbation
                # data point perturbation
                mask= torch.zeros(x_data.reshape(-1).shape, dtype=torch.bool)

                x_pert_samples= perturbation.get_perturbed_inputs(original_sample=x_data.reshape(-1),
                                                              feature_mask=mask,
                                                              num_samples=descriptor['num_samples'],
                                                              feature_metadata=descriptor['feature_metadata'])

                # --- take the closest num_perts points to x_data that have the same predicted class label to x_data
                fx_pert= model(x_pert_samples)
                y_pert_preds= torch.where(fx_pert > 0.5, 1, 0)

                # get only the first num_perts points ordered by class and distance from x_data
                x_pert_samples, y_pert_preds= self.__get_subsets(x_data.reshape(-1), y_pred.reshape(-1), 
                                                      x_pert_samples, y_pert_preds.reshape(-1), 
                                                      descriptor['num_perts'])

                # -------- explain each x_data perturbation
                exp_pert_samples= torch.zeros_like(x_pert_samples)

                x_ris_ratios= []
                x_ros_ratios= []

                # For each perturbation, calculate the explanation
                for i, x_pert in enumerate(x_pert_samples):
                    df_x_pert= pd.DataFrame(data=[x_pert.numpy()], columns=data.columns)            
                    exp_pert_samples[i, :]= self.__generate_explanations_pert(method, model, df_x_pert)

                    # --------- get stability for each explanator and x_data perturbation
                    # RIS
                    if not ('ros' in stab_type):
                        score_ris_measure= self.__ris_measure(x_data, x_pert, 
                                                 x_exp[i], exp_pert_samples[i], 
                                                 p_norm=descriptor['p_norm'], eps=descriptor['eps_norm'])            

                    # ROS
                    if not ('ris' in stab_type):
                        score_ros_measure= self.__ros_measure(fx_data, fx_pert[i], 
                                                  x_exp[i], exp_pert_samples[i], 
                                                  p_norm=descriptor['p_norm'], eps=descriptor['eps_norm'])

                    # --- stability measures for each x_data perturbation --- one processing cicle
                    # RIS
                    if not ('ros' in stab_type):
                        x_ris_ratios.append(score_ris_measure)

                    # ROS
                    if not ('ris' in stab_type):
                        x_ros_ratios.append(score_ros_measure)

                # --- append only the max/mean value related to each x_data processed
                # max and mean values
                if not ('ros' in stab_type):
                    ris_max_ratios.append(x_ris_ratios[np.argmax(x_ris_ratios)])
                    ris_mean_ratios.append(np.mean(x_ris_ratios))

                if not ('ris' in stab_type):
                    ros_max_ratios.append(x_ros_ratios[np.argmax(x_ros_ratios)])
                    ros_mean_ratios.append(np.mean(x_ros_ratios))

            if not ('ros' in stab_type):
                metric_name = 'ris'
                ris= pd.DataFrame(data= np.asarray(np.stack((ris_max_ratios, ris_mean_ratios))).T, 
                                       columns=['max','average'])
                ris.to_csv('./'+ self.directory_source + '/' + self.directory_metrics + '/' + metric_name +
                                 '/'+ method +'.csv', index=False)

            if not ('ris' in stab_type):
                metric_name = 'ros'
                ros= pd.DataFrame(data= np.asarray(np.stack((ros_max_ratios, ros_mean_ratios))).T, 
                                       columns=['max','average'])
                ros.to_csv('./'+ self.directory_source + '/' + self.directory_metrics + '/' + metric_name +
                                 '/'+ method +'.csv', index=False)
            
    
    def plot_local_view(self, method_name: str, features_names: list, selected_metrics:list=[], color_by: str = 'label'):
        
        if method_name not in self.methods_dict_param:
            raise AttributeError(f"No explanations were generated for this method: " + method_name)
            
        if not features_names:
            selected_metrics = self.features_names[:3]
        elif len(features_names) != 3: 
            raise AttributeError(f"features_names must to contain 3 features.")
        
        if not selected_metrics:
            selected_metrics = self.metrics_used[:4]
        elif len(selected_metrics) != 4: 
            raise AttributeError(f"selected_metrics must to contain 4 metrics.")
            
        df_plot = pd.DataFrame()
        for metric_name in selected_metrics:
#             print(metric_name)
            directory = './'+ self.directory_source + '/' + self.directory_metrics + '/' + metric_name + '/'
            df = pd.read_csv(directory + method_name + '.csv')
            if 'average' in df.columns:
                df_plot[metric_name] = df['average']
            else:
                df_plot[metric_name] = df['max']
                
        df_pred = pd.read_csv('./'+ self.directory_source + '/y_pred.csv')
        
        df_plot['pred'] = df_pred['pred']
        df_plot['label'] = df_pred['label']
        df_plot['is_correct'] = df_plot['label'] == df_plot['pred']
        
        df_X = pd.read_csv('./'+ self.directory_source + '/X_sample.csv')
        for feature_name in features_names:
            df_plot[feature_name] = df_X[feature_name].values
#         return df_plot

        return self.__draw_local_view(df_plot, features_names, selected_metrics, color_by)
        
        
    def plot_global_view(self, selected_methods:list=[], selected_metrics:list=[], type_chart='global'):
        
        df_plot = pd.DataFrame(columns=['method', 'metric', 'value', 'best'])
        
        if not selected_metrics:
            selected_metrics = self.metrics_used[:4]
        elif len(selected_metrics) != 4: 
            raise AttributeError(f"selected_metrics must to contain 4 metrics.")
            
        if not selected_methods:
            selected_methods = self.methods_dict_param
        
        
        for metric_name in selected_metrics:
            summary_data = pd.DataFrame(columns=['method', 'metric', 'value', 'best'])
            directory = './'+ self.directory_source + '/' + self.directory_metrics + '/' + metric_name + '/'
            for method in selected_methods:
#             for filename in os.listdir(directory):
            # Check if the file is a .csv file
#                 if filename.endswith('.csv'):
                filepath = os.path.join(directory, method + '.csv')
                # Read the CSV file into a DataFrame
                df = pd.read_csv(filepath)
                
                if 'average' in df.columns and len(df.columns) > 1:
                    df = df.drop(columns=['average']) #delete average column
                    
#                 if metric_name == 'res':
#                     print('---------- df before minmax')
#                     print(df)
                df = self.__apply_minmax_with_zero_row(df)
                
#                 if metric_name == 'res':
#                     print('---------- df after minmax')
#                     print(df)
                
                # Calculate the average and standard deviation of all values in the DataFrame
                avg = df.mean().mean()
                
#                 if metric_name == 'res':
#                     print('Mean')
#                     print(avg)
                    
                # Append the summary information
                summary_data.loc[len(summary_data.index)] = [method, metric_name, avg, 0]
                    
            # codigo antigo para setar estrela no melhor (se tivesse 2 melhores nao funcionava)
            # Find the index of the row with the maximum value in col1
#             max_idx = summary_data['value'].idxmax()
            # Set the value of col2 to 1 for the row with the maximum value in col1
#             summary_data.at[max_idx, 'best'] = 1

            array_idx_max = summary_data.loc[summary_data['value']==summary_data['value'].max()].index.values
            summary_data.loc[summary_data.index.isin(array_idx_max), 'best'] = 1

            df_plot = pd.concat([df_plot, summary_data], axis=0)
            
        df_lamp, sequence_methods = self.projection.apply_lamp(df_plot)
        df_lamp['method'] = sequence_methods
        df_plot = df_plot.merge(df_lamp, how='inner', on='method')
        
        
        if type_chart=='global':
            return self.__draw_global_view(df_plot)
        else:
            return self.__draw_global_view_rank(df_plot)
        
    def plot_distribution_importance_values(self, method_name: str, feature_name: str, maxbins=30):
        if method_name not in self.methods_dict_param:
            raise AttributeError(f"No explanations were generated for this method: " + method_name)
            
        directory = './'+ self.directory_source + '/' + self.directory_explanations + '/' + method_name + '.csv'
        df = pd.read_csv(directory)
        df['index'] = df.index
        
        X_test = self.X_sample.copy()
        X_test['index'] = X_test.index
        df_features = df.merge(X_test, how='inner', on='index')
        
        feature_param = feature_name + ':Q'
        feature_param_y = feature_name + '_y' + ':Q'
        feature_param_x = feature_name + '_x' + ':Q'

        dist = alt.Chart(df).mark_bar().encode(
            alt.X(feature_param, bin=alt.Bin(maxbins=maxbins), title='Distribution of Importance Values'),
            y='count()',
        )

        heatmap = alt.Chart(df_features).mark_rect().encode(
            alt.Y(feature_param_y, bin=alt.Bin(maxbins=maxbins), title='Feature Values'),
            alt.X(feature_param_x, bin=alt.Bin(maxbins=maxbins), title='Importance Values'),
            alt.Color('count():Q', scale=alt.Scale(scheme='greenblue')).legend(titleOrient='right'),
            tooltip=[alt.Tooltip('count()', title='count')]
        )

        return alt.hconcat( dist,
            heatmap,
        ).resolve_legend(
            color="independent",
            size="independent"
        )
        
    def plot_importances(self, method_name: str):
        
        if method_name not in self.methods_dict_param:
            raise AttributeError(f"No explanations were generated for this method: " + method_name)
            
        
        directory = './'+ self.directory_source + '/' + self.directory_explanations + '/' + method_name + '.csv'
        df = pd.read_csv(directory)
        df['index'] = df.index
        df_table = pd.melt(df, id_vars=['index'], value_vars=self.feature_names)
        

        bars_features = alt.Chart(df_table).mark_bar().encode(
            x=alt.X('mean(value):Q').title('Mean'),
            y=alt.Y('variable:N').sort('-x').title('features'),
            color=alt.condition(
                alt.datum.mean_value > 0,
                alt.value("steelblue"),  # The positive color
                alt.value("orange")  # The negative color
            )
        ).properties(
            width=300,
            height=300
        )

        manual_ordered = df_table.groupby('variable')['value'].mean().sort_values(ascending=False).index.to_numpy()

        gaussian_jitter = alt.Chart(df_table, title='').mark_circle(size=8).encode(
            x=alt.X("value:Q").title('Values'),
            y=alt.Y('variable:N', sort=manual_ordered).title('').axis(labels=False, ticks=False),
            yOffset="jitter:Q",
            color=alt.condition(
                alt.datum.value > 0,
                alt.value("steelblue"),  # The positive color
                alt.value("orange")  # The negative color
            )
        #     color=alt.Color('variable:N').legend(None)
        ).transform_calculate(
            # Generate Gaussian jitter with a Box-Muller transform
            jitter="sqrt(-2*log(random()))*cos(2*PI*random())"
        ).properties(
            width=300,
            height=300
        )

        return alt.hconcat(bars_features, (gaussian_jitter).resolve_scale(yOffset='independent'), spacing=1)
    
    def plot_compare_importances(self, list_methods: list):
        df_list = {}

        for method_name in list_methods:
            if method_name not in self.methods_dict_param:
                raise AttributeError(f"No explanations were generated for this method: " + method_name)
            
            directory = './'+ self.directory_source + '/' + self.directory_explanations + '/' + method_name + '.csv'
            df = pd.read_csv(directory)
            df['index'] = df.index
            df_table = pd.melt(df, id_vars=['index'], value_vars=self.feature_names)
            df_list[method_name] = df_table
            manual_ordered = df_table.groupby('variable')['value'].mean().sort_values(ascending=False).index.to_numpy()

        brush = alt.selection_point(encodings=['y'])
        
        charts = []
        for method_name in df_list:
            charts.append(
                    alt.Chart(df_list[method_name], title=method_name).mark_bar(cursor="pointer").encode(
                        x=alt.X('mean(value):Q').title('Mean'),
                        y=alt.Y('variable:N').title('features'),
                        color=alt.condition(
                            alt.datum.mean_value > 0,
                            alt.value("steelblue"),  # The positive color
                            alt.value("orange")  # The negative color
                        ),
                        opacity=alt.condition(brush, alt.value(1), alt.value(0.1)),
                    ).add_params(
                        brush
                    ).properties(
                        width=300,
                        height=300
                    )
            )
        return alt.hconcat(*charts)
    
    def __draw_local_view(self, df, features_names: list, selected_metrics, color_by: str):
        
        color_feature = color_by + ':N'
        
        list_feature_names = features_names[:3] #TODO reduz para 3 o numero de features
        metric_names = []
        for metric in selected_metrics:
            metric_names.append(metric + ':Q')

        jitter = [math.sqrt(-2 * math.log(random.random())) * math.cos(2 * math.pi * random.random()) for _ in range(df.shape[0])]
        df['jitter'] = jitter

        # Brush for selection
        brush = alt.selection_interval()

        # Scatter Plot
        points = alt.Chart(df).mark_circle(size=16, opacity=0.75).encode(
            x= alt.X('jitter:Q', title=None, axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),),
            color=alt.Color(color_feature),
            opacity=alt.condition(brush, alt.value(1), alt.value(0.05)),
        ).add_params(brush).properties(
            width=90,
            height=200
        )

        box = alt.Chart(df).mark_boxplot(opacity=0.7, ticks=True, outliers=False).encode()


        ########################################################################

        # Configure heatmap

        rect_heatmap = alt.Chart(df).mark_rect().encode(
            alt.X('pred:N', axis=alt.Axis(values=[0,1])).title('predicted'),
            alt.Y('label:N', axis=alt.Axis(values=[0,1])).title('atual'),
            alt.Color('count()').scale(scheme='greenblue').title('Records')
        ).transform_filter(
            brush
        ).properties(
            width=100,
            height=100
        )

        text_heatmap = rect_heatmap.mark_text(fontSize=20).encode(
            text='count()',
            color=alt.ColorValue('grey'),
        ).transform_filter(
            brush
        )

        ########################################################################


        area1 = alt.Chart(df).mark_bar(color='lightgray', opacity=0.5, thickness=100).encode(
            alt.Y('count()')
                .stack(None),
        ).add_params(
            brush
        ).properties(
            width=250,
            height=200
        )

        area2 = alt.Chart(df).mark_bar(opacity=0.5, thickness=100).encode(
            alt.Y('count()')
                .stack(None),
        ).transform_filter(
            brush
        )

        # Build chart
        return alt.vconcat(
            alt.hconcat(
                points.encode(y=alt.Y(metric_names[0])) + box.encode(alt.Y(metric_names[0])),
                points.encode(y=alt.Y(metric_names[1])) + box.encode(alt.Y(metric_names[1])),
                points.encode(y=alt.Y(metric_names[2])) + box.encode(alt.Y(metric_names[2])),
                points.encode(y=alt.Y(metric_names[3])) + box.encode(alt.Y(metric_names[3])),
                rect_heatmap + text_heatmap,
            ),
            alt.hconcat(
                area1.encode(x=alt.X(list_feature_names[0]).bin().title(list_feature_names[0])) + area2.encode(x=alt.X(list_feature_names[0]).bin()),
                area1.encode(x=alt.X(list_feature_names[1]).bin().title(list_feature_names[1])) + area2.encode(x=alt.X(list_feature_names[1]).bin()),
                area1.encode(x=alt.X(list_feature_names[2]).bin().title(list_feature_names[2])) + area2.encode(x=alt.X(list_feature_names[2]).bin()),
            )
        ).configure_view(stroke=None)
    
    
    def __draw_global_view(self, df):
        brush = alt.selection_interval(encodings=["x", "y"])

        metric_space = alt.Chart(df, title="Metric Space").mark_point(filled=False).encode(
            alt.X('Lamp_1:Q').scale(domain=(0, 1.01)).title('best ->').axis(labels=False, ticks=False),
            alt.Y('Lamp_2:Q').scale(domain=(0, 1.01)).title('best ->').axis(labels=False, ticks=False),
            opacity=alt.condition(brush, alt.value(1), alt.value(0.1)),
            tooltip=['method']
        ).add_params(
            brush
        ).properties(
            width=350,
            height=350
        )

        metric_matrix_heatmap = alt.Chart(df).mark_rect().encode(
            x=alt.X('metric:N', title=None).axis(labels=False, ticks=False),
            y=alt.Y('method:N', title='explanation method'),
            color=alt.Color('value:Q').scale(scheme='greenblue', domain=[0, 1]).title('Value Metric').legend(labelOpacity=0, labelFontSize=1, title='Metric value', titleOrient='right'),
            opacity=alt.condition(brush, alt.value(1), alt.value(0.1)),
#             tooltip=['method','metric','value']
        ).properties(
            width=50,
            height=350
        ).add_params(
            brush
        )#.transform_filter(brush)

        metric_matrix_circ = metric_matrix_heatmap.mark_point().encode(
            alt.ColorValue('white'),
            alt.ShapeValue('M0,.5L.6,.8L.5,.1L1,-.3L.3,-.4L0,-1L-.3,-.4L-1,-.3L-.5,.1L-.6,.8L0,.5Z'),
            size = alt.condition(
                alt.datum.best > 0,
                alt.value(100),
                alt.value(0)
            )
        )



        method_domain = df.method.value_counts().index.to_list()
        method_domain = sorted(method_domain)
        metric_bars = alt.Chart(df, title="Best Results").mark_bar().encode(
            alt.X('best:Q', axis=alt.Axis(orient='bottom')).scale(domain=(0, df.metric.value_counts().size)).title(''),
            alt.Y('method:N').scale(domain=method_domain).title('').axis(labels=False, ticks=False),
            opacity=alt.condition(brush, alt.value(1), alt.value(0.1))
        ).add_params(
            brush
        ).properties(
            width=200,
            height=350
        )

        return alt.hconcat(metric_space,
                    alt.hconcat(
                        alt.layer(metric_matrix_heatmap, metric_matrix_circ, data=df).facet(column='metric:N').resolve_scale(x="independent"),
                        metric_bars,
                        spacing=1
                    )
        ).configure_facet(spacing=2)
    
    def __draw_global_view_rank(self, df):
        brush = alt.selection_interval(encodings=["x", "y"])
        selection_legend = alt.selection_point(fields=['method'], bind='legend')

        metric_space = alt.Chart(df, title="Metric Space").mark_point(filled=False).encode(
            alt.X('Lamp_1:Q').scale(domain=(0, 1.01)).title('best ->').axis(labels=False, ticks=False),
            alt.Y('Lamp_2:Q').scale(domain=(0, 1.01)).title('best ->').axis(labels=False, ticks=False),
            alt.Color('method:N'),#.scale(scheme='category20c'),
            opacity=alt.condition(brush & selection_legend, alt.value(1), alt.value(0.1)),
            tooltip=['method']
        ).add_params(
            brush,
            selection_legend
        ).properties(
            width=300,
            height=300
        )

        bump_metric = alt.Chart(df).mark_line(point={"filled": True, "size": 250}, strokeWidth=5).encode(
            x = alt.X("metric:O", title="metric"),
            y="rank:O",
            color = alt.Color('method:N'),#.scale(scheme='category20c'),
            opacity=alt.condition(brush & selection_legend, alt.value(1), alt.value(0.1)),
            tooltip=['method', 'metric']
        ).transform_window(
            rank="rank()",
            sort=[alt.SortField("value", order="descending")],
            groupby=["metric"]
        ).properties(
            title="Bump Chart for Metrics",
            width=300,
            height=300,
        ).add_params(
            brush,
            selection_legend
        )

        # Configure text
        text = bump_metric.mark_text(baseline='middle').encode(
            alt.Text('rank:O', format=".0f"),
            color=alt.condition(
                alt.datum.mean_horsepower > 150,
                alt.value('black'),
                alt.value('white')
            )
        )

        return alt.hconcat(metric_space, bump_metric + text)

            
            
    def __apply_minmax_with_zero_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a row with all values set to zero, applies MinMaxScaler to the columns,
        and then removes the zero row.

        :param df: The input DataFrame.
        :return: A DataFrame with MinMaxScaler applied to the columns, without the added zero row.
        """
        # Create a row of zeros with the same columns as the DataFrame
        zero_row = pd.DataFrame([[0] * len(df.columns)], columns=df.columns)

        # Add the zero row to the DataFrame
        df_with_zero = pd.concat([zero_row, df], ignore_index=True)

        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()

        # Apply the scaler to the DataFrame including the zero row
        scaled_df = pd.DataFrame(scaler.fit_transform(df_with_zero), columns=df.columns)

        # Remove the zero row after scaling
        scaled_df = scaled_df.iloc[1:].reset_index(drop=True)

        return 1 - scaled_df
    
            
    def __create_directory(self):

        if not os.path.exists(self.directory_source):
            os.mkdir(self.directory_source)
            
    def __create_directory_explanations(self):
        
        if not os.path.exists(self.directory_source):
            raise FileNotFoundError('load_model function must be executed before generate_explanations.')
        
        directory_subdirectory = os.path.join(self.directory_source, self.directory_explanations)
        
        if not os.path.exists(directory_subdirectory):
            os.mkdir(directory_subdirectory)
            
    def __create_directory_metrics(self, metric_name=None):
        
        if not os.path.exists(self.directory_source):
            raise FileNotFoundError('load_model function must be executed before generate_metrics.')
        
        directory_subdirectory = os.path.join(self.directory_source, self.directory_metrics)
        
        if not os.path.exists(directory_subdirectory):
            os.mkdir(directory_subdirectory)
            
        if metric_name is not None:
            directory_metric = os.path.join(directory_subdirectory, metric_name)
            if not os.path.exists(directory_metric):
                os.mkdir(directory_metric)
                
                
    ############ Métodos usados pelas métricas que o Evandro codificou
                
    def __progressbar(self, it, prefix="", size:int=60, out=sys.stdout, percentual:bool=True): # Python3.6+
        count= len(it)
        start= time.time() # time estimate start

        def show(j):
            x= int(size*j/count)

            # time estimate calculation and string
            remaining= ((time.time() - start) / j) * (count - j)
            if (remaining< 0.5): remainig= 0
            mins, sec= divmod(remaining, 60) # limited to minutes
            time_str = f"{int(mins):02}:{sec:02.1f}"

            if (percentual):
                percent  = f"{((j/ count)* 100):02.1f}"

                print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {percent}% - Est wait {time_str}", 
                      end='\r', file=out, flush=True)
            else:
                print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} - Est wait {time_str}", 
                      end='\r', file=out, flush=True)

        show(0.1) # avoid div/0

        for i, item in enumerate(it):
            yield item
            show(i+1)

        print("\n", flush=True, file=out)
        
    
    # returns the Lp norm of the difference between v1 and v2.
    # normalizes the difference between v1 and v2 by v1 (adapted; Agarwal, Chirag, et al., 2022)
    def __lp_norm_dif(self, v1, v2, p_norm=2, eps=1e-6, norm:bool=True):

        # arrays can be flattened, so long as ordering is preserved
        v1_flat= np.asarray(v1).flatten()
        v2_flat= np.asarray(v2).flatten()

        dif_flat= (v1_flat - v2_flat)

        if (norm==True):
            v1_flat= self.__clip_small_values(v1_flat, eps)
            dif_flat= np.divide(dif_flat, v1_flat, out=np.zeros_like(v1_flat), where=v1_flat!=0)

        return np.linalg.norm(dif_flat, ord=p_norm)
    
    def __get_pertubation_func(self):
        # Perturbation class parameters
        perturbation_mean= 0.0
        perturbation_std= 0.01
        perturbation_flip_percentage= 0.0001

        perturbation = NormalPerturbation('tabular',
                                         mean=perturbation_mean,
                                         std_dev=perturbation_std,
                                         flip_percentage=perturbation_flip_percentage)
        return perturbation
    
    def __get_descriptor(self):
        # define a parameters' descriptor dictionary
        descriptor= dict()
        descriptor['num_samples']= 20
        descriptor['p_norm']= 2
        descriptor['num_perts']= 10
        descriptor['feature_metadata']= ['c'] * (self.X_sample.shape[1])   # c means continuous features
        descriptor['eps_norm']= 1e-6
        return descriptor
    
    def __get_subsets(self, x, x_class, dataset, dataset_class, n_elements, option:int=0):
        data_size= dataset.shape[0]

        if (data_size< n_elements):
            raise ValueError("Data size must be greater than n_elements!")
        else:
            if (option):
                # order the dataset and dataset_class by distance from x
                dataset_order, dataset_class_order= distance_ordering(dataset, dataset_class, x.unsqueeze(0))

                # get the subset with first num_perts points by the same x class
                ind_same_class= (x_class == dataset_class_order).nonzero()[:n_elements].squeeze()

                subset= torch.index_select(input=dataset_order, dim=0, index=ind_same_class)
                subset_class= torch.index_select(input=dataset_class_order, dim=0, index=ind_same_class)
            else:
                # get the subset with first num_perts points by the same x class
                ind_same_class= (x_class == dataset_class).nonzero()[:n_elements].squeeze()

                # get only the elements in dataset under y' = y
                subset= torch.index_select(input=dataset, dim=0, index=ind_same_class)
                subset_class= torch.index_select(input=dataset_class, dim=0, index=ind_same_class)


            # if there are no elements enough in dataset matching with x_class
            if (subset.shape[0]< n_elements):
                last= n_elements - subset.shape[0]

                # we complete the n_elements of subset with the first instances of the ordered dataset
                if (option):
                    if (ind_same_class.numel()== 1): # avoid a breaking when only one element matches with x_class
                        ind_same_class= torch.tensor([ind_same_class])

                    dataset_order= remove_tensor_row_by_indexset(dataset_order, ind_same_class)
                    dataset_class_order= remove_tensor_row_by_indexset(dataset_class_order, ind_same_class)

                    dataset_order= dataset_order[0:last,:]
                    dataset_class_order= dataset_class_order[0:last]

                    subset= torch.cat((subset, dataset_order))
                    subset_class= torch.cat((subset_class.reshape(-1), dataset_class_order.reshape(-1)))

                # or we complete with x and x_class
                else:
                    fill_x, fill_x_class= [], []

                    for i in range(last):
                        fill_x.append(x)
                        fill_x_class.append(x_class)

                    fill_x= torch.stack(fill_x)
                    fill_x_class= torch.stack(fill_x_class)

                    subset= torch.cat((subset, fill_x))
                    subset_class= torch.cat((subset_class.reshape(-1), fill_x_class.reshape(-1)))


            return subset, subset_class
        
    def __generate_explanations_pert(self, method, model, X_sample):
        
        explainer_attr_dict = {
            'integrated gradients' : self.explainers.lem_ig, 
            'input x gradient' : self.explainers.lem_ixg, 
            'smoothgrad' : self.explainers.lem_sg,
            'vanilla gradient' : self.explainers.lem_vg, 
            'guided backprop' : self.explainers.lem_gb,
            'occlusion' : self.explainers.lem_oc,
            'lime' : self.explainers.lem_lime,
            'kernelshap' : self.explainers.lem_ks,
            'deeplift': self.explainers.lem_dpl,
            'gradientshap': self.explainers.lem_gs
        }

        attributions_feature_importance = []
        func_explanation_method = explainer_attr_dict[method]
        params = self.methods_dict_param[method]
        for input_ in X_sample.to_numpy():
            input_ = torch.from_numpy(input_.astype(np.float32))
            input_ = Variable(torch.FloatTensor(input_), requires_grad=True)
            input_ = torch.reshape(input_, (1, X_sample.shape[1]))

            attributions, _ = func_explanation_method(self.model, input_, params)
            attributions = attributions.detach().numpy()
            attributions_feature_importance.append(attributions[0])

        feature_importance = pd.DataFrame(attributions_feature_importance, columns=self.feature_names)
        feature_importance_tensor = torch.from_numpy(feature_importance.to_numpy().astype(np.float32))
        return feature_importance_tensor
    
    # compute norm between predictions per perturbation - RIS
    def __ris_measure(self, x_data, x_pert, exp_data, exp_pert, p_norm=2, eps=1e-6):
        x_dif_norm= self.__lp_norm_dif(x_data, x_pert, p_norm=p_norm, eps=eps, norm=True)
        x_dif_norm= self.__clip_small_values(x_dif_norm, eps)
        exp_dif_norm= self.__lp_norm_dif(exp_data, exp_pert, p_norm=p_norm, eps=eps, norm=True)
        stability_measure= np.divide(exp_dif_norm, x_dif_norm, where=x_dif_norm!=0)
        return stability_measure

    # compute norm between representations - ROS
    def __ros_measure(self, fx_data, fx_pert, exp_data, exp_pert, p_norm=2, eps=1e-6):
        fx_data= fx_data.detach()
        fx_pert= fx_pert.detach()
        fx_dif_norm= self.__lp_norm_dif(fx_data, fx_pert, p_norm=p_norm, eps=eps, norm=True)
        fx_dif_norm= self.__clip_small_values(fx_dif_norm, eps)
        exp_dif_norm= self.__lp_norm_dif(exp_data, exp_pert, p_norm=p_norm, eps=eps, norm=True)
        stability_measure= np.divide(exp_dif_norm, fx_dif_norm, where=fx_dif_norm!=0)
        return stability_measure
    
    # RETURN v clipped
    def __clip_small_values(self, v, eps=1e-6):
        v_aux= v.copy()
        if (type(v_aux)== np.ndarray):
            elements= v_aux.shape[0]
            for i in range(elements):
                if (v_aux[i]< 0 and np.abs(v_aux[i])< eps):
                    v_aux[i]= -eps
                elif (v_aux[i]> 0 and v_aux[i]< eps):
                    v_aux[i]= eps
        else:
            if (v_aux< 0 and np.abs(v_aux)< eps):
                v_aux[i]= -eps
            elif (v_aux> 0 and v_aux< eps):
                v_aux= eps

        return v_aux