# PROBABILISTIC MODELING AND ANALYSIS

This repository implements and explores state-of-the-art algorithms for probabilistic modeling and analysis of (complex) systems, which are founded in fields like statistics, machine learning, and artificial inteligence. Its main purpose is knowledge dissemination in the form of toolboxes and Jupyter notebooks. In this regard, the links below display the notebooks via [*nbviewer*](https://nbviewer.jupyter.org/) to ensure a proper rendering of the formulas. Note that parts of the material links to a repository I developed during my PhD study at Aalborg University, but the material in this repo will continuously be updated. 

# Supervised learning

## Linear regression

The following tutorials are implemented in Python;

- [Linear regression](https://nbviewer.jupyter.org/github/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/Linear-regression/LinReg.ipynb). This tutorial introduces linear regression; first, from a maximum likelihood estimation (MLE) perspective, and second, from a Bayesian perspective. In both cases, the tutorial implements a selection of different learning algorithms. 

- [Linear regression - assumptions and interpretations](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Linear-regression/LinReg_assumptionsEtc.ipynb). This notebook considers and assesses the underlaying assumptions of linear regression in detail and discusses the interpretation of these models. 

- [Bayesian linear regression with Stan](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Linear-regression/exStan_BayesLinReg.ipynb). This tutorial shows how to implement Bayesian linear regression models using the probabilistic programming language [*Stan*](https://mc-stan.org/). 

- [EM for Bayesian linear regression](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Linear-regression/exEM_BayesLinReg.ipynb). This tutorial considers how the expectation maximization (EM) algorithm may be used to learn a parameter setting for a Bayesian linear regression model.

## Logistic regression

The following tutorials are implemented in Python;

- [Binary logistic regression](https://nbviewer.jupyter.org/github/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/Logistic-regression/LogReg_Bin.ipynb). This tutorial introduces binary logistic regression from a maximum likelihood estimation (MLE) perspective and shows how to include regularization in the estimation to reduce overfitting.

## Bayesian hierarchical models

The following tutorials are implemented in Python;

- [Bayesian hierarchical models with Stan](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Hierarchical-models/HierModel_OMAE2020.ipynb). This tutorial introduces how to implement Bayesian hierarchical regression models using the probabilistic programming language [*Stan*](https://mc-stan.org/) by studying the fatigue data set in Glavind et al. (2020). Moreover, the concept of Bayesian model averaging is introduced as a means for making inferences for new out-of-sample fatigue sensitive details. 

## Bayesian networks

The following toolboxes and tutorials are implemented in R;

### Toolboxes

The toolboxes are among others used in Glavind and Faber (2018), and Glavind and Faber (2020).

- [Structure learning and dynamic discretization toolbox - `sLearningAndDiscretizationTools`](https://github.com/SebastianGlavind/PhD-study/blob/master/Bayesian-networks/Toolboxes/sLearningAndDiscretizationTools.R). The toolbox is a wrapper for the [`bnlearn`](https://www.bnlearn.com/) package, which implements the underlaying score-based routines for structure learning.

- [Parameter learning toolbox - `pLearningTools`](https://github.com/SebastianGlavind/PhD-study/blob/master/Bayesian-networks/Toolboxes/pLearningTools.R). The toolbox is a wrapper for the [`bnlearn`](https://www.bnlearn.com/) package.

### Structure learning

- [Bayesian networks - structure learning and automated discretization form complete data](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Bayesian-networks/BNs_sLearn_fullyObs.ipynb). This tutorial demonstrates how to learn the graph structure and optimal discretization policy of a Bayesian network (BN) representation from complete / fully observed data using my toolbox [`sLearningAndDiscretizationTools`](https://github.com/SebastianGlavind/PhD-study/blob/master/Bayesian-networks/Toolboxes/sLearningAndDiscretizationTools.R). 

- [Bayesian networks - structure learning and automated discretization from incomplete data](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Bayesian-networks/BNs_sLearn_partiallyObs.ipynb). This tutorial demonstrates how to learn the graph structure and optimal discretization policy of a Bayesian network (BN) representation from incomplete / partially observed data using my toolbox [`sLearningAndDiscretizationTools`](https://github.com/SebastianGlavind/PhD-study/blob/master/Bayesian-networks/Toolboxes/sLearningAndDiscretizationTools.R).

### Parameter learning

- [Bayesian networks - parameter learning from complete data](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Bayesian-networks/BNs_pLearn_fullyObs.ipynb). This tutorial demonstrates how to learn the parameters of a Bayesian network (BN) representation from complete / fully observed data using my toolbox [`pLearningTools`](https://github.com/SebastianGlavind/PhD-study/blob/master/Bayesian-networks/Toolboxes/pLearningTools.R). 

- [Bayesian networks - EM for parameter learning from incomplete data](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Bayesian-networks/BNs_pLearn_EM_partiallyObs.ipynb). This tutorial demonstrates how to learn the parameters of a Bayesian network (BN) representation from incomplete / partially observed data using the expectation-maximization (EM) algorithm. An implementation of the EM algorithm is found in my toolbox [`pLearningTools`](https://github.com/SebastianGlavind/PhD-study/blob/master/Bayesian-networks/Toolboxes/pLearningTools.R). 

- [Bayesian networks - Gibbs sampling for parameter learning from incomplete data](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Bayesian-networks/BNs_pLearn_Gibbs_partiallyObs.ipynb). This tutorial demonstrates how to learn the parameters of a Bayesian network (BN) representation from incomplete / partially observed data using Gibbs sampling. The implementation makes use of some functionalities from my toolbox [`pLearningTools`](https://github.com/SebastianGlavind/PhD-study/blob/master/Bayesian-networks/Toolboxes/pLearningTools.R). 

### Inference

- [Bayesian networks - inference for discrete Bayesian networks](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Bayesian-networks/BNs_inference.ipynb). This tutorial demonstrates how to make inferences using general bn.fit objects; these objects may have be learned using the [`bnlearn`](https://www.bnlearn.com/) package alone or in combination with my toolboxes [`sLearningAndDiscretizationTools`](https://github.com/SebastianGlavind/PhD-study/blob/master/Bayesian-networks/Toolboxes/sLearningAndDiscretizationTools.R) and [ `pLearningTools`](https://github.com/SebastianGlavind/PhD-study/blob/master/Bayesian-networks/Toolboxes/pLearningTools.R), which are wrappers for the `bnlearn` package. For this tutorial we will use the inference functionalities of the `bnlearn` package, as well as the [`gRain`](http://people.math.aau.dk/~sorenh/software/gR/) package to make maximum a-posteriori inferences, as well as posterior inferences that account for parameter uncertainties. 

## Gaussian processes

The following toolboxes and tutorials are implemented in Python and R;

### Toolboxes

- [Gaussian process tools (`GPtools`)](https://github.com/SebastianGlavind/PhD-study/blob/master/Gaussian-processes/GPtools.py). This Python toolbox summarizes the algorithms used in the tutorials to perform exact GP inference.

### Tutorials

The following tutorials are implemented in Python;

- [An intuitive introduction to Gaussian processes - Gaussian processes as the generalization of Gaussian probability distributions](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Gaussian-processes/GPintro_intuitive.ipynb). In this tutorial, we will consider how to generalize the Gaussian probability distribution to infinite dimensions to approach the Gaussian process. The tutorial is intuitive in the sense that we will consider graphics instead of proofs to motivate the transition from the finite dimensional Gaussian probability distribution to the Gaussian process.

- [Gaussian process regression](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Gaussian-processes/GPR.ipynb). This tutorial introduces Gaussian process regression; first, in a single-output setting, and second, in a multi-output setting. For the single-output case, the tutorial implements a selection of different learning algorithms, and some of the capabilities of the open source software package [`GPy`](https://sheffieldml.github.io/GPy/) are demonstrated for both cases.

- Bayesian optimization using a Gaussian process prior, see the tutorials described below on [*Bayesian optimization using a Gaussian process prior*](https://nbviewer.jupyter.org/github/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/Optimization/GaussianProcessOptimization.ipynb), [*Gradient boosting regression using XGBoost*](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Gradient-boosting/GBMReg_BostonHousing.ipynb) and [*Gradient boosting classification using XGBoost*](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Gradient-boosting/GBMClas_Wine.ipynb), respectively.

## Neural networks

The following tutorials are implemented in Python;

- [Neural network regression using keras and tensorflow](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Neural-networks/NNReg_BostonHousing.ipynb). This tutorial introduces neural network regression with keras and tensorflow by considering the Boston housing data set; first, in a single-output setting; and second, in a multi-output setting. Finally, the tutorial considers hyperparameters tuning in general models using random search cross-validation. 

- [Neural network classification using keras and tensorflow](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Neural-networks/NNClas_Wine.ipynb). This tutorial introduces neural network classification with keras and tensorflow by considering the Wine recognition data set. The tutorial first study how a neural network is implemented for classification tasks and then considers how to tune hyperparameters in general models using random search cross-validation. 

## Tree-based learners

The following tutorials are implemented in Python;

- [Gradient boosting regression using XGBoost](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Gradient-boosting/GBMReg_BostonHousing.ipynb). This tutorial introduces gradient boosting regression with [`XGBoost`](https://xgboost.readthedocs.io/en/latest/) by considering the Boston housing data set. The tutorial first study how gradient boosting is implemented in a single-output setting as well as the effect of different data pre-processing steps. Then, it is shown how gradient boosting may be extended to a multi-output setting. Finally, the tutorial considers hyperparameters tuning in general models using Bayesian optimization with a Gaussian process prior based [`GPyOpt`](https://github.com/SheffieldML/GPyOpt). 
 
- [Gradient boosting classification using XGBoost](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Gradient-boosting/GBMClas_Wine.ipynb). This tutorial introduces gradient boosting classification with [`XGBoost`](https://xgboost.readthedocs.io/en/latest/) by considering the Wine recognition data set. The tutorial first study how gradient boosting is implemented as well as the effect of different data pre-processing steps. Then, the tutorial considers hyperparameters tuning in general models using Bayesian optimization with a Gaussian process prior based [`GPyOpt`](https://github.com/SheffieldML/GPyOpt). Finally, the tutorial elaborates on the feature importance functionalities of `XGBoost`.

- The reader interested in the mechanics of gradient boosting is referred to this great [blog post](https://blog.paperspace.com/gradient-boosting-for-classification/) by Vihar Kurama on the underlaying ideas and their implementation. 

# Unsupervised learning

## K-means clustering

The following tutorials are implemented in Python;

- [K-means algorithm](https://nbviewer.jupyter.org/github/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/Kmeans/Kmeans.ipynb). This tutorial considers how the K-means algorithm may be used to find a given number latent clusters in a data set, as well as how to choose an appropriate number of clusters to consider for the data set.

## Gaussian mixture models

The following tutorials are implemented in Python;

- [EM for Gaussian mixtures](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Gaussian-mixtures/exEM_GMMs.ipynb). This tutorial considers how Gaussian mixture models may be used for cluster analysis; it implements the expectation maximization (EM) learning algorithm, and introduces the evidence lower bound, as well as the Bayesian information criterion (BIC) and the integrated complete-data likelihood (ICL), for model selection.

# Time series models

## State space models

The following toolboxes and tutorials are implemented in Python and R;

### Toolboxes

- [Dynamic linear modeling tools (DLMtools)](https://github.com/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/State-space-models/DLMtools.py). The Python toolbox collects the methods and algorithms used in the tutorials. Furthermore, this [*notebook*](https://github.com/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/State-space-models/SSMs_linearGaussian_Ex_MassSpringSys_testTB.ipynb) shows how we can use the toolbox to solve one of the examples in my tutorial on [*Linear Gaussian state space models and Kalman filters*](https://nbviewer.jupyter.org/github/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/State-space-models/SSMs_linearGaussian_Ex_MassSpringSys.ipynb), and this [*notebook*](https://github.com/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/State-space-models/SSMs_linearGaussian_Ex6_5.ipynb) shows how to solve an example from Shumway & Stoffer (2017).

### Tutorials

- [Linear Gaussian state space models and Kalman filters](https://nbviewer.jupyter.org/github/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/State-space-models/SSMs_linearGaussian_Ex_MassSpringSys.ipynb). This Python tutorial implements the Kalman filtering and smooting equations for Linear Gaussian state space models. Furthermore, the tutorial shows how to learn the parameters of such systems using maximum likelihood and Bayesian inference based on data (accelerations) from simple mass-spring systems. Finally, this [*R-notebook*](https://github.com/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/State-space-models/SSMs_linearGaussian_Ex_MassSpringSys_R_dlm.ipynb) shows how one of the examples can be solved using the R package `dlm`.

- [Model uncertainty in Linear Gaussian state space models](https://nbviewer.jupyter.org/github/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/State-space-models/SSMs_linearGaussian_Ex_MassSpringSys_ModUnc.ipynb). This Python tutorial consideres the mass-spring-damper system from my tutorial [*Linear Gaussian state space models and Kalman filters*](https://nbviewer.jupyter.org/github/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/State-space-models/SSMs_linearGaussian_Ex_MassSpringSys.ipynb). For this system, it shows how we may accound for uncertainty in the system representation itself and the loading using Monte Carlo simulation and Bayesian inference.

# Inference

## Algorithms for optimization

The following tutorials are implemented in Python;

- [Deterministic algorithms for unconstrained, continuous-valued optimization](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Optimization/Optimization_deter_con_uncon.ipynb). This tutorial considers set of local derivative-based optimization algorithms for unconstrained, continuous-valued optimization. The algorithms covered are first-order methods, i.e., gradient decent and its variations (e.g., conjugate gradient decent and Adam), and second-order methods, i.e. Newton's method and quasi-Newton methods (DFP and BFGS).

- [Stochastic algorithms for unconstrained, continuous-valued optimization](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Optimization/Optimization_stoch_con_uncon.ipynb). This tutorial considers set of stochastic optimization algorithms, including population methods, for unconstrained, continuous-valued optimization. The algorithms covered are stochastic gradient decent, stochastic hill-climbing, simulated annealing, genetic algorithms, and particle swarm optimization.

- [Bayesian optimization using a Gaussian process prior](https://nbviewer.jupyter.org/github/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/Optimization/GaussianProcessOptimization.ipynb). This tutorial shows how to perform Bayesian optimization using a Gaussian process prior for unconstrained, continuous-valued optimization. The tutorial make use of some of the capabilities of the open source software package [`GPy`](https://sheffieldml.github.io/GPy/) for parameter learning in both a maximum likelihood setting and a Bayesian setting.

## Approximate inference

The following tutorials are implemented in Python;

### Monte Carlo methods

#### Toolboxes

- [Subset simulation tools (`SuStools`)](https://github.com/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/Approximate-inference/SuStools.py). This Python toolbox collects the methods and algorithms needed to implement subset implementation with adaptive conditional sampling (Papaioannou et al., 2015).

#### Tutorials

- [Metropolis-Hastings algorithm - a tutorial introduction](https://nbviewer.jupyter.org/github/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/Approximate-inference/MCMC_MHintro.ipynb). This tutorial introduces and implements the fundamental Metropolis-Hastings alogrithm, as well as its independent-component variant, which is more suitable for high-dimentional problems.

- [Change of variables in MCMC simulations](https://nbviewer.jupyter.org/github/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/Approximate-inference/MCMC_ChangeOfVariables.ipynb). This tutorial shows how the variables in MCMC simulations can be transformed to comply with limiting supports, without introducing biases in the simulation. The key element here is the determinant of the Jacobian matrix for the transformation, which we defines for a set of common support limits and their combination.

- [Hamiltonian Monte carlo - a basic tutorial](https://nbviewer.jupyter.org/github/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/Approximate-inference/HMC.ipynb). This tutorial introduces the ideas behind the Hamiltonian Monte Carlo alogrithm and implement some basic variants of the algorithm, which are showcased in a linear regression setting.

- [Subset simulation for rare event sampling](https://nbviewer.jupyter.org/github/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/Approximate-inference/SuS_aCS.ipynb). This tutorial introduces the ideas behind subset simulation for rare/extreme event sampling and solves some baseline examples from the literature using my subset simulation toolbox [`SuStools`](https://github.com/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/Approximate-inference/SuStools.py).

- *Approximate Bayesian computation (ABC) - an introduction*. COMMING SOON

# Additional topics

## Sensitivity analysis and feature selection

The following tutorials are implemented in Python;

- [Variance-based sensitivity analysis for independent inputs](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Sensitivity-analysis/SA_varianceBased_independentInputs.ipynb). This tutorial implements a set of methods, which are applicable when the inputs are independent. First, a surrogate-based method is considered that decomposes the variance based on linear regression considerations. Second, two simulation-based methods are introduced; the first method performs conditional sampling by binning the input space, and the second method performs efficient conditional sampling.  

- [Variance-based sensitivity analysis for correlated inputs](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Sensitivity-analysis/SA_varianceBased_correlatedInputs.ipynb). This tutorial implements a set of methods, which are applicable when the inputs are correlated. First, two surrogate-based methods are considered; the first method decomposed the variance based on (linear) regression considerations, and the second method decomposes the variance based on a polynomial chaos expansion. Second, two simulation-based methods are introduced; the first method performs conditional sampling by binning the input space, and the second method performs conditional sampling for randomly sampled input realizations.  

## Hyperparameter tuning, model selection and automated machine learning (AutoML)

- Random search for hyperparameter tuning, see the tutorials described above on [*Neural network regression using keras and tensorflow*](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Neural-networks/NNReg_BostonHousing.ipynb) and [*Neural network classification using keras and tensorflow*](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Neural-networks/NNClas_Wine.ipynb), respectively.

- Bayesian optimization using a Gaussian process prior for hyperparameter tuning, see the tutorials described above on [*Gradient boosting regression using XGBoost*](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Gradient-boosting/GBMReg_BostonHousing.ipynb) and [*Gradient boosting classification using XGBoost*](https://nbviewer.jupyter.org/github/SebastianGlavind/PhD-study/blob/master/Gradient-boosting/GBMClas_Wine.ipynb), respectively.

- Bayesian calibration of simulators. The following tutorials show to perform Baysian calibration for a SDOF mass-spring-damper system model using [*basic MCMC inference*](https://nbviewer.jupyter.org/github/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/Tuning-selection/BayesianCalibration.ipynb), [*ABC rejection sampling*](https://nbviewer.jupyter.org/github/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/Tuning-selection/BayesianCalibration_ABCrejection.ipynb), and [*ABC subset simulation*](https://nbviewer.jupyter.org/github/SebastianGlavind/Probabilistic_modeling_and_analysis/blob/main/Tuning-selection/BayesianCalibration_ABCsubset.ipynb), respectively.

# References
<!-- *** -->
Sebastian T. Glavind, Henning Brüske and Michael H. Faber, “On normalized fatigue crack growth modeling”, in proceedings of the ASME 2020 39th International Conference on Ocean, Offshore and Arctic Engineering (OMAE2020), OMAE2020-18613, 2020.

Sebastian T. Glavind and Michael H. Faber, “A framework for offshore load environment modeling”, Journal of Offshore Mechanics and Arctic Engineering, vol. 142, no. 2,
pp. 021702, OMAE-19-1059, 2020.

Sebastian T. Glavind and Michael H. Faber, “A framework for offshore load environment modeling”, in proceedings of the ASME 2018 37th International Conference on Ocean, Offshore and Arctic Engineering (OMAE2018), OMAE2018-77674, 2018.

Shumway, R. H. & Stoffer, D. S. (2017). Time series analysis and its applications, Springer.

Papaioannou, I., Betz, W., Zwirglmaier, K., & Straub, D. (2015). MCMC algorithms for subset simulation. Probabilistic Engineering Mechanics, 41, 89-103.
<!-- *** -->

