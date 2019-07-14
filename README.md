# TreeExplainer

Package for explaining and interpreting predictions of tree-based machine learning models. The notion of interpretability is based on how close the inclusion of a feature takes the model toward its final prediction. For this reason, the result of this approach is "feature contributions" to the predictions. 

The basic idea is to decompose each prediction into feature contribution components. For a dataset with ``n`` features, each prediction on the dataset is calculated as

prediction ≈ baseline probability at tree root + **contribution feature 1** + ... + **contribution feature** ***n***

Feature contributions are provided at the level of observations, features, and targets.

## Implementation

The class TreeExplainer encapsulates the tree-based model and accepts data on which to calculate feature contributions. It also offers several methods to generate reports, graphs and useful metrics. In particular, TreeExplainer offers the following:

+ Compute feature contributions on new data, for example, held-out data for validation or testing. [Other implementations do not offer this feature]
+ All operations performed on individual trees can be run in parallel.
+ Feature contributions are correctly computed when the same feature is used more than once along a branch of a decision tree. 
+ Non-informative nodes can be discarded from the final result. This may happen because splitting on a certain value in training set resulted in some information gain, but now it doesn't for test samples.
+ Support for pandas DataFrames.
+ Methods to generate high-quality plots.
+ Summaries are generated to be easily visualized in notebooks for exploratory data analyses.
+ More specialized computations are performed as needed.
+ Borrows ideas from established R packages.


## Supported models

This module is under active development. Currently, the following models have been tested:
    
+ Random forests
    + sklearn.ensemble.forest.RandomForestClassifier
        
+ Decision trees
    + sklearn.tree.tree.DecisionTreeClassifier

In the near future, all ensemble methods in scikit-learn will be tested and supported.

## Support for pandas and non-numerical data

TreeExplainer fully supports data (and even targets) contained in pandas DataFrames. Categorical data (as specified by [pandas.CategoricalDataType](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html)) are welcome, too.

Column names are stored internally to be used in plots and reports. 

Non-numerical data is internally encoded to numerical either via one-hot encoding or simply converted to numbers. The user can set a threhsold on feature cardinality to prefer either encoding method. In fact, above a certain number of categories, the sparsity introduced by one-hot encoding would deteriorate model's performance and should not be used.

    
## Dependencies

The class TreeExplainer depends on:

+ [scikit-learn](https://scikit-learn.org/) for modeling and validation
+ [numpy](https://www.numpy.org/) to calculate feature contributions
+ [pandas](https://pandas.pydata.org/) to generate tables and reports
+ [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) to generate graphs
+ [scipy](https://www.scipy.org/) for inferential statistics
+ [joblib](https://joblib.readthedocs.io/en/latest/) to process trees in parallel


# Usage

Simply import the class and pass your trained model and some test data to it (either a numpy array or a pandas DataFrame):

    from treeexplainer import TreeExplainer
    TE = TreeExplainer(model, X_test)
    
Available methods are

    .explain_single_prediction()
    .compute_min_depth_distribution()
    .compute_two_way_interactions()
    .summarize_importance()

    .plot_min_depth_distribution()
    .plot_two_way_interactions()
    
The [classification tutorial notebook](notebooks/classification_tutorial.ipynb) (currently undocumented, but stay tuned for updates.) shows basic and advanced methods of the class.


# Disclaimer and license

This core code to compute feature contributions is based on the Github repo [andosa/treeinterpreter](https://github.com/andosa/treeinterpreter/), released under BSD license. An indepedent implementation is also available in the package [ELI5](https://eli5.readthedocs.io/en/latest/autodocs/eli5.html#eli5.explain_prediction).

TreeExplainer is distributed under MIT license.

## Why TreeExplainer?

TreeExplainer is a complete revamp of the original work above, which provides more flexibility, thorough documentation for further development, and advanced features ported to python from R packages. 

On the other hand, if you only need feature contribution values, 
treeinterpreter is generally faster because it doesn't calculate anything else. 

ELI5's implementation to compute feature contribution offers a beautiful output and supports other methods for interpretability. However, it lacks a method to automatically loop through a large dataset without creating its own, complex objects.

For a comparison benchmark between these implementations, please refer to the [benchmark notebook](notebooks/benchmark.ipynb)

# References
## Papers

+ The idea has been originally introduced for regression models by [Kuz’min *et al*](https://doi.org/10.1002/minf.201000173) on Molecular Informatics in 2011.
+ It was later generalized to classification models by [Palczewska *et al*](https://arxiv.org/abs/1312.1121) in 2013.

## Blog posts

+ To compute feature contributions, a proof-of-concept implementation in python is available at [andosa/treeinterpreter](https://github.com/andosa/treeinterpreter/), which the author introduced and discussed in:
    + [Interpreting random forests](http://blog.datadive.net/interpreting-random-forests/) on 19 Oct 2014
    + [Random forest interpretation with scikit-learn](http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/) on 12 Aug 2015
    + [Random forest interpretation – conditional feature contributions](http://blog.datadive.net/random-forest-interpretation-conditional-feature-contributions/) on 24 Oct 2016 

---
# To-do
+ Implement statistical and visualization features from R package [randomForestExplainer](https://mi2datalab.github.io/randomForestExplainer/), which are described [here for regression models](https://rawgit.com/MI2DataLab/randomForestExplainer/master/inst/doc/randomForestExplainer.html) and [here for classification tasks](https://rawgit.com/geneticsMiNIng/BlackBoxOpener/master/randomForestExplainer/inst/doc/randomForestExplainer.html).
+ Integrate [random-forest-importances](https://github.com/parrt/random-forest-importances) to compute drop-column and permutation importance.
+ Add agreement plots.
+ Write notebooks to describe class features.
+ Write notebooks for benchmarks versus treeexplainer and ELI5.

