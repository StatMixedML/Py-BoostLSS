# Py-BoostLSS: An extension of Py-Boost to probabilistic modelling

We present a probabilistic extension of the recently introduced [Py-Boost](https://github.com/sb-ai-lab/Py-Boost) approach. Modelling and predicting the entire conditional distribution of univariate and multivariate responses greatly improves upon existing tree-based gradient boosting implementations, as it allows to create probabilistic predictions from which intervals and quantiles of interest can be derived. 

## Motivation

Existing implementations of Gradient Boosting Machines, such as XGBoost and LightGBM, are mostly designed for single-target regression tasks. While efficient for low to medium target-dimensions, the computational cost of estimating these models becomes prohibitive in high-dimensional settings. As an example, consider modelling a multivariate Gaussian distribution with `D=100` target variables, where the covariance matrix is approximated using the Cholesky-Decomposition. Modelling all conditional moments requires estimation of `D(D + 3)/2 = 5,150` parameters. Because both XGBoost and LightGBM follow a *one vs. all strategy*, with a separate tree grown for
each parameter, estimating these many parameters for a large dataset can become computationally very expensive. The recently introduced [Py-Boost](https://github.com/sb-ai-lab/Py-Boost) model provides a more runtime efficient GBM implementation, which makes it a good candidate for efficiently estimating high-dimensional target variables in a probabilistic setting.

Borrowing from the original paper [SketchBoost: Fast Gradient Boosted Decision Tree for Multioutput Problems](https://openreview.net/forum?id=WSxarC8t-T), the following figure illustrates the runtime-efficiency of the Py-Boost Model.

<p align="center">
<img src="https://user-images.githubusercontent.com/41187941/205011855-0e06247f-609f-4c12-9c53-9e00df91b2d9.png" width="350" height="200" />
</p>

Even though the original implementation of Py-Boost also supports estimation of univariate responses, Py-BoostLSS focuses on multi-target probabilistic regression settings. For univariate probabilistic GBMs, we refer to our implementations of [XGBoostLSS](https://github.com/StatMixedML/XGBoostLSS) and [LightGBMLSS](https://github.com/StatMixedML/LightGBMLSS).

## Acknowledgements

The implementation of Py-BoostLSS relies on the following resources:

- [Py-boost: a research tool for exploring GBDTs](https://github.com/sb-ai-lab/Py-Boost)
- [SketchBoost: Fast Gradient Boosted Decision Tree for Multioutput Problems](https://openreview.net/forum?id=WSxarC8t-T)

We genuinely thank the original authors [Anton Vakhrushev](https://www.kaggle.com/btbpanda) and [Leonid Iosipoi](http://iosipoi.com/) for making their work publicly available. 

## Reference Paper
März, Alexander (2022) [*Multi-Target XGBoostLSS Regression*](https://arxiv.org/abs/2210.06831). <br/>
März, A. and Kneib, T. (2022) [*"Distributional Gradient Boosting Machines"*](https://arxiv.org/abs/2204.00778). <br/>
März, Alexander (2019) [*XGBoostLSS - An extension of XGBoost to probabilistic forecasting*](https://arxiv.org/abs/1907.03178).
