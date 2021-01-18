# Unsupervised Ensemble Learning with Dependent Classifiers
Repository containing all the codes developed and used during my final bachelor's degree thesis (TFG), with the title *Unsupervised Ensemble Learning with Dependent Classifiers*.

All files are written in MATLAB R2020b. The file names containing '_main' are key functions that cluster and make use of other minor functions. The files are structured as follows:

* *method1_main.m*: generates correlated binary data using Method 1
  * *compute_prob.m*: generates the probabilities assigned to each set of classification results as in Function 1
  * *allocate_prob.m*: generates correlated sets of classification results by allocating the previous probabilities as in Function 2
* *method2_main.m*: generates correlated binary data using Method 2 (summarized in Function 3)
* *detect_structure_main.m*: detects the classifiers' correlation structure as in Algorithm 1
  * *estimate_rank_1_matrix.m*: estimates diagonal values of a rank-one matrix
  * *tracemin.m*: estimates unknown entries of a rank-one matrix via trace minimization (IMPORTANT NOTE: this file makes use of the SDPT3 (version 4.0) solver, which may be found at https://github.com/Kim-ChuanToh/SDPT3)
  * *scorematrix.m*: computes the score matrix as in Equation 1.12
* *CEM_main.m*: provides ensemble estimates of ground truth labels using the CEM algorithm as summarized in Algorithm 2
  * *assignfun.m*: computes the assignment function ***c*** from the indicator function (Equation 1.9)
* *LSM_main.m*: provides ensemble estimates of ground truth labels using the LSM algorithm as summarized in Algorithm 3
  * *assignfun.m*: (already listed)
  * *estimate_ensemble_parameters.m*: estimates the ensemble's specificities and sensitivities, as well as the vector containing the weights used to compute the LSM-based ground truth labels
  * *estimate_class_imbalance_restricted_likelihood.m*: estimates the instances' class imbalance (Equation 1.2) 
  * *estimate_rank_1_matrix.m*: (already listed)
  * *tracemin.m*: (already listed)
