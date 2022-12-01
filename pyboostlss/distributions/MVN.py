import torch
from torch.autograd import grad as autograd
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import multivariate_normal
import cupy as cp 
import numpy as np



########################################################################################################################
###############################################      Multivariate Normal      ##########################################
########################################################################################################################

class MVN: 
    """Multivariate Normal Distribution Class

    """
    
    # @staticmethod
    def initialize(self, y_true: cp.ndarray, n_target: int) -> cp.ndarray:
        """ Function that calculates the starting values, for each distributional parameter individually.
        y: cp.ndarray
            Data from which starting values are calculated.
        n_target: ndarray
            Number of target variables
        """
        # Indices
        tril_indices = cp.asarray(cp.tril_indices(n_target))

        # Target
        target = y_true[:,:n_target]

        # Location
        loc_init = cp.mean(target,axis=0) 

        # Tril
        tril_init = cp.cov(target,rowvar=False)
        tril_init = cp.linalg.cholesky(tril_init)
        cp.fill_diagonal(tril_init, cp.log(cp.diagonal(tril_init))) 
        tril_init = tril_init[tril_indices[0], tril_indices[1]]
        start_values = cp.concatenate([loc_init, tril_init])

        return start_values



    # @staticmethod
    def n_dist_param(self, n_targets: int) -> int:
        """Infers the number of distributional parameters from target dimension.
        """
        n_param = int((n_targets*(n_targets + 3))/2)
        
        return n_param



    # @staticmethod
    def response_dim(self, n_param: int) -> int:
        """Infers the number of targets from number of distributional parameters.
        """    
        a = 1
        b = 3
        c = -2*n_param
        d = (b**2) - (4*a*c)
        n_target = int((-b+cp.sqrt(d))/(2*a))

        return n_target
    
    
    # @staticmethod
    def tril_dim(self, n_target: int) -> int:
        """Infers the number of lower diagonal elements from number of targets
        """    
        n_tril = int((n_target * (n_target + 1)) / 2)

        return n_tril
    
    
    # @staticmethod
    def calc_corr(self, cov_mat: torch.tensor) -> torch.tensor:
        """Calculates the lower correlation matrix from covariance matrix.
        """   
        diag = torch.sqrt(torch.diag(torch.diag(cov_mat)))
        diag_inv = torch.linalg.inv(diag)
        cor_mat = diag_inv @ cov_mat @ diag_inv 
        cor_mat = cor_mat[np.tril_indices_from(cor_mat, k=-1)]

        return cor_mat
    
    
    
    # @staticmethod
    def identity(self, predt: torch.tensor) -> torch.tensor:
        """Identity mapping of predt.
        """
        return predt
    
    
    # @staticmethod
    def exp_fun(self, predt: torch.tensor) -> torch.tensor:
        """Exp() function used to ensure predt is strictly positive.
        """
        predt_adj = torch.exp(predt) 
        predt_adj = torch.nan_to_num(predt_adj, nan=float(torch.nanmean(predt_adj))) + torch.tensor(1e-6, dtype=predt_adj.dtype, device="cuda")
        
        return predt_adj
    
    
    
    # @staticmethod
    def create_param_dict(self, n_target, tril_indices):
        """ Dictionary that holds the name of distributional parameter and their corresponding response functions.
        """

        n_theta = self.n_dist_param(n_target)    
        n_tril = self.tril_dim(n_target)


        # Location
        param_dict = {"location_" + str(i+1): self.identity for i in range(n_target)}

        # Tril
        tril_idx = (tril_indices.detach().numpy()) + 1
        tril_indices_row = tril_idx[0]
        tril_indices_col = tril_idx[1]
        tril_diag = tril_idx[0] == tril_idx[1]

        tril_dict = {}

        for i in range(n_tril):
            if tril_diag[i] == True:
                tril_dict.update({"scale_" + str(tril_idx[:,i][1]): self.exp_fun}) 
            else:
                tril_dict.update({"rho_" + str(tril_idx[:,i][0]) + str(tril_idx[:,i][1]): self.identity})

        param_dict.update(tril_dict)

        return param_dict    
    
    
    
    # @staticmethod
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
                tril_dict.update({"scale_" + str(tril_idx[:,i][1]): self.exp_fun})
            else:
                tril_dict.update({"rho_" + str(tril_idx[:,i][0]) + str(tril_idx[:,i][1]): self.identity})

        return tril_dict
    
    
    # @staticmethod
    def get_target_params_nll(self, y_true: cp.ndarray, y_pred: cp.ndarray, requires_grad=False) -> torch.tensor:
        """ Returns target and estimated parameters, where parameters are transformed to the output scale
        
        Args:
            y_true: cp.ndarray, Input target variables
            y_pred: cp.ndarray, predictions
            requires_grad: bool(), Whether or not tensor requires gradient for automatic differentiation
            
        Returns:
        target, predt, predt_location, predt_tril, nll
        """
        
        ###
        # Initialize 
        ###
        n_obs = y_true.shape[0]
        n_param = y_true.shape[1]
        n_target = self.response_dim(n_param)                                            
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

        # Location
        predt_location = torch.concat(predt[:n_target],axis=1)    

        # Tril: response function has to be included in computational graph explicitly
        preds_scale = self.exp_fun(predt[1])
        tril_predt = predt[n_target:]
        tril_predt = [response_fun(tril_predt[i]) for i, (dist_param, response_fun) in enumerate(tril_param_dict.items())] 
        tril_predt = torch.concat(tril_predt,axis=1)    
        predt_tril = torch.zeros(n_obs, n_target, n_target, dtype=tril_predt.dtype, device="cuda")
        predt_tril[:, tril_indices[0], tril_indices[1]] = tril_predt
        
        
        ###
        # NLL
        ###
        dist_fit = MultivariateNormal(loc=predt_location, scale_tril=predt_tril)   
        nll = -torch.nansum(dist_fit.log_prob(target)) 
        
        return target, predt, predt_location, predt_tril, nll
    
    
    
    
    # @staticmethod
    def get_derivs(self, nll: torch.tensor, predt: torch.tensor) -> cp.ndarray:
        """ Calculates gradients and hessians.
        
        Output gradients and hessians have shape (n_samples, n_outputs).
        
        Args:
            nll: torch.tensor, calculated NLL
            predt: torch.tensor, list of predicted paramters 
            
        Returns:
        grad, hess
        """
        
        # Gradient and Hessian
        grad_list = autograd(nll, inputs=predt, create_graph=True)
        hess_list = [autograd(grad_list[i].nansum(), inputs=predt[i], retain_graph=True)[0] for i in range(len(grad_list))]         
        
        # Reshape
        grad = cp.asarray(torch.concat(grad_list,axis=1).detach())
        hess = cp.asarray(torch.concat(hess_list,axis=1).detach())      
        
        return grad, hess
    

    
    
    


#     ###
#     # Function for drawing random samples from predicted distribution
#     ###
#     def pred_dist_rvs(pred_params: pd.DataFrame, n_samples: int, seed: int):
#         """
#         Function that draws n_samples from a predicted response distribution.

#         pred_params: pd.DataFrame
#             Dataframe with predicted distributional parameters.
#         n_samples: int
#             Number of sample to draw from predicted response distribution.
#         seed: int
#             Manual seed.
#         Returns
#         -------
#         pd.DataFrame with n_samples drawn from predicted response distribution.

#         """
#         pred_dist_list = []

#         for i in range(pred_params.shape[0]):
#             pred_dist_list.append(norm.rvs(loc=pred_params.loc[i,"location"],
#                                            scale=pred_params.loc[i,"scale"],
#                                            size=n_samples,
#                                            random_state=seed)
#                                   )

#         pred_dist = pd.DataFrame(pred_dist_list)
#         return pred_dist

#     ###
#     # Function for calculating quantiles from predicted distribution
#     ###
#     def pred_dist_quantile(quantiles: list, pred_params: pd.DataFrame):
#         """
#         Function that calculates the quantiles from the predicted response distribution.

#         quantiles: list
#             Which quantiles to calculate
#         pred_params: pd.DataFrame
#             Dataframe with predicted distributional parameters.

#         Returns
#         -------
#         pd.DataFrame with calculated quantiles.

#         """
#         pred_quantiles_list = []

#         for i in range(len(quantiles)):
#             pred_quantiles_list.append(norm.ppf(quantiles[i],
#                                                 loc = pred_params["location"],
#                                                 scale = pred_params["scale"])
#                                        )

#         pred_quantiles = pd.DataFrame(pred_quantiles_list).T
#         return pred_quantiles
