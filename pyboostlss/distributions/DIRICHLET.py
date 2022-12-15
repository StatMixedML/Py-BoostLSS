import torch
from torch.distributions.dirichlet import Dirichlet
from dirichlet.dirichlet import mle as dirichlet_mle
import cupy as cp
import numpy as np
import pandas as pd
from pyboostlss.utils import * 



########################################################################################################################
###############################################      Dirichlet Distribution    #########################################
########################################################################################################################

class DIRICHLET:
    """Dirichlet Distribution Class
    """
    
    def __init__(self, D:int):
        self.D = D # specifies target dimension


    def initialize(self, y_true: cp.ndarray, n_target: int) -> cp.ndarray:
        """ Function that calculates the starting values, for each distributional parameter individually.
        y_true: cp.ndarray
            Data from which starting values are calculated.
        n_target: ndarray
            Number of target variables
        """

        start_values = np.log(dirichlet_mle(cp.asnumpy(y_true)))

        return cp.array(start_values)


    def n_dist_param(self, n_targets: int) -> int:
        """Infers the number of distributional parameters from target dimension.
        """

        return n_targets



    def target_append(self, y_true: np.ndarray, n_param: int) -> np.ndarray:
        """Function that appends target to the number of specified parameters
        """

        return cp.array(y_true)




    def create_param_dict(self, n_target):
        """ Dictionary that holds the name of distributional parameter and their corresponding response functions.
        """

        # Alpha
        param_dict = {"alpha_" + str(i+1): exp_fn for i in range(n_target)}

        return param_dict    
    

    
    

    def get_params_nll(self, y_true: cp.ndarray, y_pred: cp.ndarray, requires_grad=False) -> torch.tensor:
        """ Returns estimated parameters and nll.
        
        Args:
            y_true: cp.ndarray, Input target variables
            y_pred: cp.ndarray, predictions
            requires_grad: bool(), Whether or not tensor requires gradient for automatic differentiation
            
        Returns:
        predt, nll
        """
        
        ###
        # Initialize 
        ###
        n_target = n_param = self.D
        param_dict = self.create_param_dict(n_target)


        ###
        # Target
        ###    
        target = torch.as_tensor(y_true, device="cuda").reshape(-1, n_target)



        ###
        # Parameters
        ###   
        predt = [
            torch.tensor(
                y_pred[:,i].reshape(-1,1), device="cuda", requires_grad=requires_grad
            ) for i in range(n_param)
        ]

        # Alpha
        predt_alpha = torch.concat(
            [response_fun(predt[i]) for i, (dist_param, response_fun) in enumerate(param_dict.items())],
            axis=1
        )

        
        
        ###
        # NLL
        ###
        dist_fit = Dirichlet(predt_alpha)
        nll = -torch.nansum(dist_fit.log_prob(target)) 
        
        return predt, nll
    
    
    
    
    def predict(self, 
                model,  
                X_test: np.array, 
                pred_type: str = "parameters",                    
                n_samples: int = 100
               ):
        """
        Predict function.

        model: 
            Instance of pyboostlss
        X_test: np.array
            Test data features
        pred_type: str
            Specifies what is to be predicted:
                "samples": draws n_samples from the predicted response distribution. Output shape is (n_samples, n_obs, n_target)
                "parameters": returns the predicted distributional parameters.
        n_samples: int
            If pred_type="response" specifies how many samples are drawn from the predicted response distribution.
        Returns
        -------
        pd.DataFrame with n_samples drawn from predicted response distribution.

        """
         
        n_target = self.D
        param_dict = self.create_param_dict(n_target)
        dist_params = list(param_dict.keys())
        
        # Predicted parameters
        params_predt = torch.tensor(model.predict(X_test), device="cuda")
        params_predt = torch.cat(
            [response_fun(params_predt[:, i]).reshape(-1,1) for i, (dist_param, response_fun) in enumerate(param_dict.items())],
            axis=1)

        # Predicted Distribution
        dirichlet_pred = Dirichlet(params_predt)

        # Output DataFrame
        predt_params = pd.DataFrame(params_predt.cpu().detach().numpy(),columns=dist_params)
        
        if pred_type == "parameters":
            return predt_params        
        
        elif pred_type == "samples":
            torch.manual_seed(123)
            y_samples = dirichlet_pred.sample((n_samples,)).cpu().detach().numpy()
            return y_samples
