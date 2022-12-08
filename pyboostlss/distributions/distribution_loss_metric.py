import cupy as cp 
from pyboostlss.utils import *
from py_boost.gpu.losses import Loss, Metric



class Distribution_Metric(Metric):
    
    
    def __init__(self, dist):
        self.dist = dist
            
    
    alias = "NLL-score"
    

    def error(self, y_true, y_pred):
        """Error metric definition. 
        Args:
            y_true: cp.array, targets
            y_pred: cp.array, predictions
            sample_weight: None or cp.ndarray, weights
        Returns:
            float, metric value
        """  

        _, nll = self.dist.get_params_nll(y_true, y_pred)        
        nll = cp.asarray(nll)

        return nll


    def compare(self, v0 ,v1):
        """
        It should return True if v0 metric value is better than v1, False othewise
        """        
        return v0 < v1
    

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Full metric definition. 
        Args:
            y_true: cp.array, targets
            y_pred: cp.array, predictions
            sample_weight: None or cp.ndarray, weights
        Returns:
            float, metric value
        """
        
        err = self.error(y_true, y_pred)
        
        return err
    
    
    
    
class Distribution_Loss(Loss):
    
    def __init__(self, dist):
        self.dist = dist
    
    def get_grad_hess(self, y_true, y_pred):
        """
        Defines how to calculate gradients and hessians for given loss.
        Args:
            y_true: cp.array, targets
            y_pred: cp.array, predictions
            sample_weight: None or cp.ndarray, weights
        Returns:
            floats, grad, hess
        """         
        
        ###
        # Parameters and NLL
        ###
        predt, nll = self.dist.get_params_nll(y_true, y_pred, requires_grad=True)
        
        
        ###
        # Derivatives
        ###   
        grad, hess = get_derivs(nll, predt)       
        
        return grad, hess
    
    
    
    def base_score(self, y_true):
        """
        Defines how parameter estimates are initialized. 
        Args:
            y_true: cp.array, targets
        Returns:
            floats, base_margins            
        """             

        n_target = self.dist.D
        base_margin = self.dist.initialize(y_true, n_target)
                      
        return base_margin