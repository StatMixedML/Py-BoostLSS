import torch
from torch.autograd import grad as autograd
import cupy as cp 
import numpy as np

###
# Response Functions
###

def identity_fn(predt: torch.tensor) -> torch.tensor:
    """Identity mapping of predt.
    """
    return predt


def exp_fn(predt: torch.tensor) -> torch.tensor:
    """Exp() function used to ensure predt is strictly positive.
    """
    predt_adj = torch.exp(predt) 
    predt_adj = torch.nan_to_num(predt_adj, nan=float(torch.nanmean(predt_adj))) + torch.tensor(1e-6, dtype=predt_adj.dtype, device="cuda")

    return predt_adj




###
# Autograd Function 
###
def get_derivs(nll: torch.tensor, predt: torch.tensor) -> cp.ndarray:
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




###
# Misc
###

def response_dim(y_true: int) -> int:
    """Infers the number of targets from input dataset.
    """    
    n_obs = y_true.shape[0]
    col_sums = y_true.sum(axis=0)
    n_target = col_sums != n_obs
    n_target = len(n_target[n_target == True])

    return n_target


def calc_corr(cov_mat: torch.tensor) -> torch.tensor:
    """Calculates the lower correlation matrix from covariance matrix.
    """   
    diag = torch.sqrt(torch.diag(torch.diag(cov_mat)))
    diag_inv = torch.linalg.inv(diag)
    cor_mat = diag_inv @ cov_mat @ diag_inv 
    cor_mat = cor_mat[np.tril_indices_from(cor_mat, k=-1)]

    return cor_mat



