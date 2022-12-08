import torch
from pyro.distributions import MultivariateStudentT
from scipy.stats import t
import cupy as cp 
import numpy as np
import pandas as pd
from pyboostlss.utils import *  



########################################################################################################################
###############################################      Multivariate Student-T   ##########################################
########################################################################################################################

class MVT: 
    """Multivariate Student-T Distribution Class, where covariance matrix \Sigma is estimated via Cholesky-decomposition
    \Sigma = LL`.
    """
    
    def __init__(self,D:int):
        self.D = D # specifies target dimension
        

    def initialize(self, y_true: cp.ndarray, n_target: int) -> cp.ndarray:
        """ Function that calculates the starting values, for each distributional parameter individually.
        y_true: cp.ndarray
            Data from which starting values are calculated.
        n_target: ndarray
            Number of target variables
        """
        # Target
        target = y_true[:,:n_target]
        
        # Fitted Student-T Parameters 
        student_param = cp.array([t.fit(cp.asnumpy(target[:,i])) for i in range(n_target)])
        
        # Df 
        df_init = (cp.log(2) + cp.log(cp.median(student_param[:,0]))).reshape(-1,)
        
        # Location
        loc_init = cp.mean(target,axis=0) 

        # Tril
        tril_indices = cp.asarray(np.tril_indices(n_target))
        tril_init = cp.cov(target,rowvar=False)
        tril_init = cp.linalg.cholesky(tril_init)
        cp.fill_diagonal(tril_init, cp.log(cp.diagonal(tril_init))) 
        tril_init = tril_init[tril_indices[0], tril_indices[1]]
        
        start_values = cp.concatenate([df_init, loc_init, tril_init])

        return start_values



    def n_dist_param(self, n_targets: int) -> int:
        """Infers the number of distributional parameters from target dimension.
        """
        n_param = int(1 + ((n_targets*(n_targets + 3))/2))
        
        return n_param


    def target_append(self, y_true: np.ndarray, n_param: int) -> np.ndarray:
        """Function that appends target to the number of specified parameters
        """
        n_obs = y_true.shape[0]
        n_target = y_true.shape[1]
        n_fill = n_param - n_target
        np_fill = np.ones((n_obs, n_fill))
        y_append = np.concatenate([y_true, np_fill],axis=1)

        return y_append


    def tril_dim(self, n_target: int) -> int:
        """Infers the number of lower diagonal elements from number of targets.
        """    
        n_tril = int((n_target * (n_target + 1)) / 2)

        return n_tril
    
    
    def rho_dim(self, n_target: int) -> int:
        """Infers the number of correlations from number of targets.
        """           
        n_rho = int((n_target * (n_target - 1)) / 2)
        return n_rho
    
    

    def create_param_dict(self, n_target, tril_indices):
        """ Dictionary that holds the name of distributional parameter and their corresponding response functions.
        """

        n_theta = self.n_dist_param(n_target)    
        n_tril = self.tril_dim(n_target)
        
        # Df
        param_dict = {"df": exp_fn}    

        # Location
        loc_dict = {"location_" + str(i+1): identity_fn for i in range(n_target)}
        
        param_dict.update(loc_dict)

        # Tril
        tril_idx = (tril_indices.detach().numpy()) + 1
        tril_indices_row = tril_idx[0]
        tril_indices_col = tril_idx[1]
        tril_diag = tril_idx[0] == tril_idx[1]

        tril_dict = {}

        for i in range(n_tril):
            if tril_diag[i] == True:
                tril_dict.update({"scale_" + str(tril_idx[:,i][1]): exp_fn}) 
            else:
                tril_dict.update({"rho_" + str(tril_idx[:,i][0]) + str(tril_idx[:,i][1]): identity_fn})

        param_dict.update(tril_dict)

        return param_dict      
    

    def create_tril_dict(self, n_target, tril_indices):
        """ Dictionary that holds the name of distributional parameter and their corresponding response functions.
        """

        n_theta = self.n_dist_param(n_target)    
        n_tril = self.tril_dim(n_target)

        # Tril
        tril_idx = (tril_indices.detach().numpy()) + 1
        tril_indices_row = tril_idx[0]
        tril_indices_col = tril_idx[1]
        tril_diag = tril_idx[0] == tril_idx[1]

        tril_dict = {}

        for i in range(n_tril):
            if tril_diag[i] == True:
                tril_dict.update({"scale_" + str(tril_idx[:,i][1]): exp_fn})
            else:
                tril_dict.update({"rho_" + str(tril_idx[:,i][0]) + str(tril_idx[:,i][1]): identity_fn})

        return tril_dict   
    

    def get_params_nll(self, y_true: cp.ndarray, y_pred: cp.ndarray, requires_grad=False) -> torch.tensor:
        """ Returns target and estimated parameters, where parameters are transformed to the output scale
        
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
        n_obs = y_true.shape[0]
        n_param = y_true.shape[1]
        n_target = self.D                                       
        n_tril = self.tril_dim(n_target)
        tril_indices = torch.tril_indices(row=n_target, col=n_target, offset=0)       
        param_dict = self.create_param_dict(n_target,tril_indices)                       
        tril_param_dict = self.create_tril_dict(n_target,tril_indices)     


        ###
        # Target
        ###    
        target = torch.as_tensor(y_true[:,:n_target], device="cuda").reshape(-1, n_target)   



        ###
        # Parameters
        ###   
        predt = [torch.tensor(y_pred[:,i].reshape(-1,1), device="cuda", requires_grad=requires_grad) for i in range(n_param)]
        
        # Df
        predt_df = exp_fn(predt[0]).reshape(-1,) 

        # Location
        predt_location = torch.concat(predt[1:(n_target+1)],axis=1)     

        # Tril: response function has to be included in computational graph explicitly
        tril_predt = predt[(n_target+1):]
        tril_predt = [response_fun(tril_predt[i]) for i, (dist_param, response_fun) in enumerate(tril_param_dict.items())] 
        tril_predt = torch.concat(tril_predt,axis=1)    
        predt_tril = torch.zeros(n_obs, n_target, n_target, dtype=tril_predt.dtype, device="cuda")
        predt_tril[:, tril_indices[0], tril_indices[1]] = tril_predt
        
        
        ###
        # NLL
        ###
        dist_fit = MultivariateStudentT(predt_df, predt_location, predt_tril)
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
        n_tril = self.tril_dim(n_target)
        n_rho = self.rho_dim(n_target)
        tril_indices = torch.tril_indices(row=n_target, col=n_target, offset=0)       
        param_dict = self.create_param_dict(n_target,tril_indices)  
        dist_params = list(param_dict.keys())
        
        # Predicted parameters
        params_predt = torch.tensor(model.predict(X_test), device="cuda")
        params_predt = [response_fun(params_predt[:, i]).reshape(-1,1) for i, (dist_param, response_fun) in enumerate(param_dict.items())]
        
        # Df
        predt_df = params_predt[0].reshape(-1,)
        predt_df_pd = pd.DataFrame(predt_df.cpu().detach().numpy())
        predt_df_pd.columns = ["df"]        

        # Location
        predt_location = torch.concatenate(params_predt[1:(n_target+1)],axis=1)
        predt_location_df = pd.DataFrame(predt_location.cpu().detach().numpy())
        predt_location_df.columns = [param for param in dist_params if "location_" in param]

        # Tril
        n_obs = X_test.shape[0]
        tril_predt = torch.concatenate(params_predt[(n_target+1):],axis=1).reshape(-1, n_tril)
        predt_tril = torch.zeros(n_obs, n_target, n_target, dtype=tril_predt.dtype, device="cuda")
        predt_tril[:, tril_indices[0], tril_indices[1]] = tril_predt                

        # Predicted Distribution
        mvt_pred = MultivariateStudentT(predt_df, predt_location, predt_tril)

        # Sigma
        predt_sigma = mvt_pred.stddev.cpu().detach().numpy()
        predt_sigma_df = pd.DataFrame(predt_sigma)
        predt_sigma_df.columns = [param for param in dist_params if "scale_" in param]

        # Rho
        cov_mat = mvt_pred.covariance_matrix
        predt_rho = torch.concatenate([calc_corr(cov_mat[i]).reshape(-1, n_rho) for i in range(n_obs)],axis=0)
        predt_rho_df = pd.DataFrame(predt_rho.cpu().detach().numpy())
        predt_rho_df.columns = [param for param in dist_params if "rho_" in param]   
        

        # Output DataFrame
        predt_params = pd.concat([predt_df_pd, predt_location_df, predt_sigma_df, predt_rho_df], axis=1)  
        
        if pred_type == "parameters":
            return predt_params        
        
        elif pred_type == "samples":
            torch.manual_seed(123)
            y_samples = mvt_pred.sample((n_samples,)).cpu().detach().numpy()
            return y_samples
        
