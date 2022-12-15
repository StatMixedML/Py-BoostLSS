import torch
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
import torch.optim as optim
import cupy as cp 
import numpy as np
import pandas as pd
from pyboostlss.utils import * 



########################################################################################################################
###############################################      Multivariate Normal      ##########################################
########################################################################################################################

class MVN_LRA: 
    """Multivariate Normal Distribution Class, where covariance matrix \Sigma is estimated via LRA apprixmation.
    """
    
    def __init__(self, 
                 r:int,
                 D:int):
        self.r = r # specifies rank
        self.D = D # specifies target dimension
        self.dtype = torch.float32
        
        
        
    def initialize(self, y_true: cp.ndarray, n_target: list) -> cp.ndarray:    
        """ Function that calculates the starting values, for each distributional parameter individually. It uses the L-BFGS algorithm for estimating unconditional parameter estimates.
    
        y_true: cp.ndarray
                Data from which starting values are calculated.
        n_target: list
                List that holds number of targets and rank-parameter.
        """

        torch.manual_seed(123)

        n_param = self.n_dist_param(n_target)
        n_target = self.D
        param_init = torch.ones(1, n_param, device="cuda", dtype=self.dtype)
        param_init = torch.nn.init.xavier_uniform_(param_init)    
        param_init.requires_grad=True
        y_true_tens = torch.tensor(y_true[:,:n_target], device="cuda", dtype=self.dtype)


        def nll_init(y_true: cp.ndarray, y_pred: cp.ndarray, requires_grad=True) -> torch.tensor:        

            n_target = self.D  
            n_param = self.n_dist_param([self.D, self.r])
            rank = self.r

            ###
            # Target
            ###    
            target = torch.as_tensor(y_true[:,:n_target], device="cuda", dtype=self.dtype).reshape(-1, n_target)

            ###
            # Parameters
            ###
            predt = [y_pred[:, i].reshape(-1,1) for i in range(n_param)]

            # Location
            predt_location = torch.concat(predt[:n_target],axis=1)

            # Low Rank Factor
            predt_covfactor = torch.concat(predt[n_target:(n_param-n_target)], axis=1).reshape(-1, n_target, rank) # (n_obs, n_target, rank)

            # Low Rank Diagonal (must be positive)
            predt_covdiag = predt[-n_target:]
            predt_covdiag = [exp_fn(predt_covdiag[i]) for i in range(len(predt_covdiag))]
            predt_covdiag = torch.concat(predt_covdiag, axis=1)

            ###
            # NLL
            ###
            dist_fit = LowRankMultivariateNormal(loc=predt_location, cov_factor=predt_covfactor, cov_diag=predt_covdiag, validate_args=False)   
            nll = -torch.nansum(dist_fit.log_prob(target)) 

            return nll

    
        def closure():
            
            lbfgs.zero_grad()
            objective = nll_init(y_true=y_true_tens, y_pred=param_init)
            objective.backward()

            return objective



        lbfgs = optim.LBFGS(params=[param_init],
                            lr=1e-03,
                            history_size=10, 
                            max_iter=4, 
                            line_search_fn="strong_wolfe")

        for i in range(20):
            lbfgs.step(closure)

        start_values = cp.array(lbfgs.param_groups[0]["params"][0].cpu().detach()).reshape(-1,)

        return start_values   
    
    
        

#     def initialize(self, y_true: cp.ndarray, n_target: list) -> cp.ndarray:
#         """ Function that initializes each distributional parameter with ones. Compared to the LBFGS, this is more runtime efficient.
#         y_true: cp.ndarray
#             Data from which starting values are calculated.
#         n_target: list
#             List that holds number of targets and rank-parameter.
#         """
#         n_param = self.n_dist_param(n_target)
#         start_values = cp.ones((n_param,))

#         return start_values




    def n_dist_param(self, n_targets: list) -> int:
        """Number of distributional parameters.
        """
        n_param = int(n_targets[0]*(2+n_targets[1]))
        
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


    def create_param_dict(self, n_target):
        """ Dictionary that holds the name of distributional parameter and their corresponding response functions.
        """
        n_target = self.D
        rank = self.r

        # Location
        param_dict = {"location_" + str(i+1): identity_fn for i in range(n_target)}

        # Low Rank Factor
        lrf_dict = {"lrf_" + str(i+1): identity_fn for i in range(n_target*rank)}
        param_dict.update(lrf_dict)

        # Low Rank Diagonal
        lrd_dict = {"lrd_" + str(i+1): exp_fn for i in range(n_target)}
        param_dict.update(lrd_dict)

        return param_dict


    def param_names(self, n_target):
        """ List that holds the name of distributional parameter.
        """

        n_tril = self.tril_dim(n_target)

        # Location
        param_names = ["location_" + str(i+1) for i in range(n_target)]

        # Tril
        tril_indices = torch.tril_indices(row=n_target, col=n_target, offset=0)
        tril_idx = (tril_indices.detach().numpy()) + 1
        tril_indices_row = tril_idx[0]
        tril_indices_col = tril_idx[1]
        tril_diag = tril_idx[0] == tril_idx[1]

        for i in range(n_tril):
            if tril_diag[i] == True:
                param_names.append("scale_" + str(tril_idx[:,i][1]))
            else:
                param_names.append("rho_" + str(tril_idx[:,i][0]) + str(tril_idx[:,i][1]))

        return param_names




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
        n_obs = y_true.shape[0]
        n_param = y_true.shape[1]
        n_target = self.D                                       
        rank = self.r
        param_dict = self.create_param_dict(n_target)


        ###
        # Target
        ###    
        target = torch.as_tensor(y_true[:,:n_target], device="cuda", dtype=self.dtype).reshape(-1, n_target)


        ###
        # Parameters
        ###
        predt = [torch.tensor(np.nan_to_num(y_pred[:, i], nan=float(np.nanmean(y_pred[:, i]))), device="cuda", requires_grad=requires_grad, dtype=self.dtype).reshape(-1,1) for i in range(n_param)]

        # Location
        predt_location = torch.concat(predt[:n_target],axis=1)

        # Low Rank Factor
        predt_covfactor = torch.concat(predt[n_target:(n_param-n_target)], axis=1).reshape(-1, n_target, rank) # (n_obs, n_target, rank)

        # Low Rank Diagonal (must be positive)
        predt_covdiag = predt[-n_target:]
        predt_covdiag = [exp_fn(predt_covdiag[i]) for i in range(len(predt_covdiag))]
        predt_covdiag = torch.concat(predt_covdiag, axis=1)


        ###
        # NLL
        ###
        dist_fit = LowRankMultivariateNormal(loc=predt_location, cov_factor=predt_covfactor, cov_diag=predt_covdiag, validate_args=False)   
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

        n_obs = X_test.shape[0]
        n_target = self.D
        rank = self.r
        n_param = self.n_dist_param([n_target, rank])
        n_rho = self.rho_dim(n_target)
        param_dict = self.create_param_dict(n_target)
        dist_params = self.param_names(n_target)
        
        # Predicted parameters
        params_predt = torch.tensor(model.predict(X_test), device="cuda")
        params_predt = [response_fun(params_predt[:, i]).reshape(-1,1) for i, (dist_param, response_fun) in enumerate(param_dict.items())]


        # Location
        predt_location = torch.cat(params_predt[:n_target],axis=1)
        predt_location_df = pd.DataFrame(predt_location.cpu().detach().numpy())
        predt_location_df.columns = [param for param in dist_params if "location_" in param]

        # Low Rank Factor
        predt_covfactor = torch.cat(params_predt[n_target:(n_param-n_target)], axis=1).reshape(-1, n_target, rank) # (n_obs, n_target, rank)
        
        # Low Rank Diagonal
        predt_covdiag = torch.cat(params_predt[-n_target:], axis=1)

        # Predicted Distribution
        mvn_lra_pred = LowRankMultivariateNormal(loc=predt_location, cov_factor=predt_covfactor, cov_diag=predt_covdiag, validate_args=False)   

        # Sigma
        predt_sigma = mvn_lra_pred.stddev.cpu().detach().numpy()
        predt_sigma_df = pd.DataFrame(predt_sigma)
        predt_sigma_df.columns = [param for param in dist_params if "scale_" in param]

        # Rho
        cov_mat = mvn_lra_pred.covariance_matrix
        predt_rho = torch.cat([calc_corr(cov_mat[i]).reshape(-1, n_rho) for i in range(n_obs)],axis=0)
        predt_rho_df = pd.DataFrame(predt_rho.cpu().detach().numpy())
        predt_rho_df.columns = [param for param in dist_params if "rho_" in param]

        # Output DataFrame
        params_df = pd.concat([predt_location_df, predt_sigma_df, predt_rho_df], axis=1)
        
        if pred_type == "parameters":
            return params_df
        
        elif pred_type == "samples":
            torch.manual_seed(123)
            y_samples = mvn_lra_pred.sample((n_samples,)).cpu().detach().numpy()
            return y_samples
