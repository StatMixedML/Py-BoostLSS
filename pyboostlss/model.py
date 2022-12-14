from pyboostlss.distributions.distribution_loss_metric import *
from pyboostlss.utils import *
from py_boost import SketchBoost

import optuna
from optuna.samplers import TPESampler

class PyBoostLSS:
    """
    Py-BoostLSS model class. Currently only supports SketchBoost algorithm.
    
    """

    def __init__(self, dist):
        self.dist = dist # pyboostlss.distributions class. Specifies distribution


    def train(self,
              dtrain=None,
              eval_sets=None, 
              ntrees=100,
              lr=0.05,
              min_gain_to_split=0,
              lambda_l2=1,
              gd_steps=1,
              max_depth=6,
              min_data_in_leaf=10,
              colsample=1.,
              subsample=1.,

              quantization='Quantile',
              quant_sample=2000000,
              max_bin=256,
              min_data_in_bin=3,

              es=100,
              seed=123,
              verbose=10,

              sketch_outputs=1,
              sketch_method="proj",
              use_hess=True,

              callbacks=None,
              sketch_params=None):
        
        """Train a pyboostlss model with given parameters.

        Parameters
        ----------
        dtrain: dict, Dataset used for training of the form {'X': X_train, 'y': X_train}
        eval_sets: list used to evaluate model during training, e.g., [{'X': X_train, 'y': X_train}]
        ntrees: int, maximum number of trees
        lr: float, learning rate
        min_gain_to_split: float >=0, minimal gain to split
        lambda_l2: float > 0, l2 leaf regularization
        gd_steps: int > 0, number of gradient steps
        max_depth: int > 0, maximum tree depth. Setting it to large values (>12) may cause OOM for wide datasets
        min_data_in_leaf: int, minimal leaf size. Note - for some loss fn leaf size is approximated
            with hessian values to speed up training
        colsample: float or Callable, sumsample of columns to construct trees or callable - custom sampling
        subsample: float or Callable, sumsample of rows to construct trees or callable - custom sampling
        quantization: str or Quantizer, method for quantizatrion. One of 'Quantile', 'Uniform',
            'Uniquant' or custom implementation
        quant_sample: int, subsample to quantize features
        max_bin: int in [2, 256] maximum number of bins to quantize features
        min_data_in_bin: int in [2, 256] minimal bin size. NOTE: currently ignored
        es: int, early stopping rounds. If 0, no early stopping
        seed: int, random state
        verbose: int, verbosity freq
        sketch_outputs: int, number of outputs to keep
        sketch_method: str, name of the sketching strategy. Currently the following options are available: "topk", "rand", "proj". 
        use_hess: bool, use hessians in multioutput training
        callbacks: list of Callback, callbacks to customize training are passed here
        sketch_params: dict, optional kwargs for sketching strategy
        
        """
        
        bstLSS_init = SketchBoost(loss=Distribution_Loss(self.dist),
                                  metric=Distribution_Metric(self.dist),
                                  ntrees=ntrees,
                                  lr=lr,
                                  min_gain_to_split=min_gain_to_split,
                                  lambda_l2=lambda_l2,
                                  gd_steps=gd_steps,
                                  max_depth=max_depth,
                                  min_data_in_leaf=min_data_in_leaf,
                                  colsample=colsample,
                                  subsample=subsample,

                                  quantization=quantization,
                                  quant_sample=quant_sample,
                                  max_bin=max_bin,
                                  min_data_in_bin=min_data_in_bin,

                                  es=es,
                                  seed=seed,
                                  verbose=verbose,

                                  sketch_outputs=sketch_outputs,
                                  sketch_method=sketch_method,
                                  use_hess=use_hess,

                                  callbacks=callbacks,
                                  sketch_params=sketch_params
                                  )
        
        
        # Append Target
        if hasattr(self.dist, "r"):
            n_target = [self.dist.D, self.dist.r]
        else:
            n_target = self.dist.D            
        
        y_train_append = self.dist.target_append(dtrain["y"], self.dist.n_dist_param(n_target))
        
        if eval_sets is not None:
            y_eval_append = self.dist.target_append(eval_sets[0]["y"] , self.dist.n_dist_param(n_target))
            eval_sets_append = eval_sets.copy()
            eval_sets_append[0]["y"] = y_eval_append
            
        else:
            eval_sets_append = None       
            
        
        bstLSS_train = bstLSS_init.fit(dtrain["X"], y_train_append, eval_sets=eval_sets_append)
        
        return bstLSS_train
            
                




    def hyper_opt(self,
                  params=None,
                  dtrain=None,
                  eval_sets=None, 
                  ntrees=100,
                  lr=0.05,
                  min_gain_to_split=0,
                  lambda_l2=1,
                  gd_steps=1,
                  max_depth=6,
                  min_data_in_leaf=10,
                  colsample=1.,
                  subsample=1.,

                  quantization='Quantile',
                  quant_sample=2000000,
                  max_bin=256,
                  min_data_in_bin=3,

                  es=100,
                  seed=123,
                  hp_seed=None,
                  verbose=int(1e04),

                  sketch_outputs=1,
                  sketch_method="proj",
                  use_hess=True,

                  callbacks=None,
                  sketch_params=None,
                  
                  max_minutes=120,           
                  n_trials=None,             
                  study_name=None,
                  silence=False
                 ):
        
        """Function to tune hyper-parameters using Optuna.
        
        Parameters
        ----------
        params: dict, tunable hyper-parameters and their ranges
        dtrain: dict, Dataset used for training of the form {'X': X_train, 'y': X_train}
        eval_sets: list used to evaluate model during training, e.g., [{'X': X_train, 'y': X_train}]
        ntrees: int, maximum number of trees
        lr: float, learning rate
        min_gain_to_split: float >=0, minimal gain to split
        lambda_l2: float > 0, l2 leaf regularization
        gd_steps: int > 0, number of gradient steps
        max_depth: int > 0, maximum tree depth. Setting it to large values (>12) may cause OOM for wide datasets
        min_data_in_leaf: int, minimal leaf size. Note - for some loss fn leaf size is approximated
            with hessian values to speed up training
        colsample: float or Callable, sumsample of columns to construct trees or callable - custom sampling
        subsample: float or Callable, sumsample of rows to construct trees or callable - custom sampling
        quantization: str or Quantizer, method for quantizatrion. One of 'Quantile', 'Uniform',
            'Uniquant' or custom implementation
        quant_sample: int, subsample to quantize features
        max_bin: int in [2, 256] maximum number of bins to quantize features
        min_data_in_bin: int in [2, 256] minimal bin size. NOTE: currently ignored
        es: int, early stopping rounds. If 0, no early stopping
        seed: int, random state
        hp_seed: int, Random state for random number generator used in the Bayesian hyper-parameter search
        verbose: int, verbosity freq
        sketch_outputs: int, number of outputs to keep
        sketch_method: str, name of the sketching strategy. Currently the following options are available: "topk", "rand", "proj". 
        use_hess: bool, use hessians in multioutput training
        callbacks: list of Callback, callbacks to customize training are passed here
        sketch_params: dict, optional kwargs for sketching strategy
        max_minutes: int, Time budget in minutes, i.e., stop study after the given number of minutes.
        n_trials: int, The number of trials. If this argument is set to None, there is no limitation on the number of trials.
        study_name : str, Name of the hyperparameter study.
        silence: bool, Controls the verbosity of the trail, i.e., user can silence the outputs of the trail.

        Returns
        -------
        opt_params : Dict() with optimal parameters.
        """

        def objective(trial):
            
            hyper_params = {
                "lr": trial.suggest_float("lr", params["lr"][0], params["lr"][1]),
                "max_depth": trial.suggest_int("max_depth", params["max_depth"][0], params["max_depth"][1]),
                "sketch_outputs": trial.suggest_int("sketch_outputs", params["sketch_outputs"][0], params["sketch_outputs"][1]),
                "lambda_l2": trial.suggest_float("lambda_l2", params["lambda_l2"][0], params["lambda_l2"][1]),
                "colsample": trial.suggest_float("colsample", params["colsample"][0], params["colsample"][1]),
                "subsample": trial.suggest_float("subsample", params["subsample"][0], params["subsample"][1]),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", params["min_gain_to_split"][0], params["min_gain_to_split"][1])
            }   
            
            bstLSS_cv = self.train(dtrain=dtrain,
                                   eval_sets=eval_sets,
                                   ntrees=ntrees,
                                   lr=hyper_params["lr"],
                                   min_gain_to_split=hyper_params["min_gain_to_split"],
                                   lambda_l2=hyper_params["lambda_l2"],
                                   gd_steps=gd_steps,
                                   max_depth=hyper_params["max_depth"],
                                   min_data_in_leaf=min_data_in_leaf,
                                   colsample=hyper_params["colsample"],
                                   subsample=hyper_params["subsample"],

                                   quantization=quantization,
                                   quant_sample=quant_sample,
                                   max_bin=max_bin,
                                   min_data_in_bin=min_data_in_bin,

                                   es=es,
                                   seed=seed,
                                   verbose=verbose,

                                   sketch_outputs=hyper_params["sketch_outputs"],
                                   sketch_method=sketch_method,
                                   use_hess=use_hess,

                                   callbacks=callbacks,
                                   sketch_params=sketch_params
                                   )


            # Add optimal rounds
            opt_rounds = bstLSS_cv.best_round
            trial.set_user_attr("opt_round", int(opt_rounds))

            # Extract the best score
            y_true = eval_sets[0]["y"]
            y_pred = bstLSS_cv.predict(eval_sets[0]["X"])
            _, nll = self.dist.get_params_nll(y_true, y_pred)
            best_score = cp.asarray(nll)

            # Replace 0 value 
            best_score = cp.where(best_score == -0.0, 1e08, best_score)

            return best_score
        
        
        if silence:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
        if study_name is None:
            study_name = "Py-BoostLSS Hyper-Parameter Optimization"
    
        if hp_seed is not None:
            sampler = TPESampler(seed=hp_seed) 
        else:
            sampler = TPESampler()            
        
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize", study_name=study_name)
        study.optimize(objective, n_trials=n_trials, timeout=60 * max_minutes, show_progress_bar=True)


        print("\nHyper-Parameter Optimization successfully finished.")
        print("  Number of finished trials: ", len(study.trials))
        print("  Best trial:")
        opt_param = study.best_trial

        # Add optimal stopping round
        opt_param.params["opt_rounds"] = study.trials_dataframe()["user_attrs_opt_round"][
            study.trials_dataframe()["value"].idxmin()]
        opt_param.params["opt_rounds"] = int(opt_param.params["opt_rounds"])

        print("    Value: {}".format(opt_param.value))
        print("    Params: ")
        for key, value in opt_param.params.items():
            print("    {}: {}".format(key, value))

        return opt_param.params
        