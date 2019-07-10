# Standard
import threading
from warnings import warn

# Numerical
import numpy as np
import pandas as pd
from scipy.stats import binom_test
from joblib import Parallel, delayed

# Graphical
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Local repo
from utilities.preprocessing import DataProcessor
from utilities.model_validation import validate_model_type, validate_model_is_trained
from utilities.parallel import compute_tree_paths
from utilities.parallel import compute_feature_contributions_from_tree
from utilities.parallel import compute_two_way_conditional_contributions
from utilities.parallel import compute_explanation_of_prediction
from tree_explainer.utilities.numerical import divide0
from tree_explainer.utilities.visualization import adjust_spines
from tree_explainer.utilities.lists import true_list


################################################################################
class TreeExplainer(object):
    def __init__(self, model, n_jobs=-1, verbose=False):
        """The class is instantiated by passing model to train and training data.
        Optionally, feature names and target names can be passed, too.

        :param model: The input model to explain.
        :param n_jobs: [int or None] The number of parallel processes to use.
        :param verbose: [bool] Whether to print progress to the console.
        """

        # Check that model type is supported
        model_class, model_type, implementation, estimator_type = validate_model_type(model)
        # Check that model has been fit, and get information from it
        n_estimators_attr, estimators_attr = validate_model_is_trained(model, model_class)

        if verbose:
            print('Analyzing model structure ...')
        # Keep a reference to these attributes of the model
        self._internals = dict(model_class=model_class,
                               model_type=model_type,
                               implementation=implementation,
                               estimator_type=estimator_type,
                               estimators_=estimators_attr,
                               n_estimators=n_estimators_attr)
        # Initialize basic attributes of the model
        self.model = model
        self.n_trees = getattr(model, self._internals['n_estimators'])
        self.n_features = model.n_features_

        # Extract decision path of each tree in model
        results = dict(feature_names=np.arange(self.n_features))
        results['tree_path'] = list(np.empty((self.n_trees, ), dtype=object))
        results['features_split'] = list(np.empty((self.n_trees, ), dtype=object))
        results['tree_feature_path'] = list(np.empty((self.n_trees,), dtype=object))
        results['tree_depth'] = np.zeros((self.n_trees, ), dtype=int)
        results['feature_depth'] = {f: np.zeros((self.n_trees, ), dtype=int) - 1 for f in results['feature_names']}
        results['threshold_value'] = list(np.empty((self.n_trees,), dtype=object))
        results['no_of_nodes'] = {f: np.zeros((self.n_trees,), dtype=int) - 1 for f in results['feature_names']}
        # Process trees in parallel
        Parallel(n_jobs=n_jobs, verbose=verbose, require='sharedmem')(
                delayed(compute_tree_paths)(
                        estimator=estimator, i_tree=i_tree,
                        results=results, lock=threading.Lock())
                for i_tree, estimator in enumerate(getattr(model, self._internals['estimators_'])))

        # Store results
        self.tree_path = results['tree_path']
        self.features_split = results['features_split']
        self.tree_feature_path = results['tree_feature_path']
        self.tree_depth = results['tree_depth']
        self.feature_depth = results['feature_depth']
        self.threshold_value = results['threshold_value']
        self.no_of_nodes = results['no_of_nodes']

        # Initialize all other TreeExplainer's attributes. Initialized explicitly
        # to silence warnings of interpreters.
        self.data = None
        self.targets = None
        self.n_samples = None
        self.predictions = None
        self.correct_predictions = None
        self.contributions = None
        self.conditional_contributions = None
        self.feature_combinations = None
        self.conditional_contributions_sample = None
        self.data_leaves = None
        self.min_depth_frame = None
        self.min_depth_frame_summary = None
        self.importance_frame = None
        self.two_way_contribution_table = None
        self.n_two_way_contribution_table = None

        if verbose:
            print('done')


    def explain_features(self, data, targets=None, n_jobs=-1, verbose=False):
        """Main method to explain the provided predictions of the model. It
        computes feature contributions, such that:
        predictions ≈ target_frequency_at_root + feature_contributions.

        REFERENCE: This function is based on _predict_forest in the python
        package treeinterpreter.
        SOURCE: https://github.com/andosa/treeinterpreter/blob/master/treeinterpreter/treeinterpreter.py

        :param data: [numpy array or pandas DataFrame] Data on which to test
            feature contributions. It must have the same number of features of
            the dataset used to train the model. It should be scaled in the same
            way as the training data.
        :param targets: [numpy array] True target values of each instance in
            `data`. For classification tasks, it contains the class labels, for
            regression problems the true value.
        :param n_jobs: [int or None] The number of parallel processes to use if
            joblib is installed. If None, trees are processed sequentially.
        :param verbose: [bool] Whether to print messages to the console regarding
            progress and outcomes.

        :return self: This allows the user to call this method together with
            initialization, and return the object in a variable, that is,
            TE = TreeExplainer(model).explain(X_test)

        The following attributes are stored in self:
        target_frequency_at_root: [numpy array] Contains the baseline prediction
            of each feature to each observation and class, averaged across trees.
        contributions: [numpy array] Contains the contribution of each feature
            to each observation and class, averaged across trees.
        """

        # Prepare data
        if verbose:
            print('Pre-processing data ...')
        self._process_and_predict(data=data, targets=targets, calling_method='explain_features')

        if verbose:
            print('Computing feature contributions ...')
        # Compute contributions
        results = dict()
        results['data_leaves'] = np.zeros((self.n_samples, self.n_trees), dtype=int)
        results['tree_path'] = list(np.empty((self.n_trees,), dtype=object))
        results['contributions'] = np.zeros((self.n_samples, self.n_features, self.n_target_levels), dtype=np.float32)
        results['contributions_n_evaluations'] = np.zeros((self.n_samples, self.n_features, self.n_target_levels), dtype=np.float32)
        # Process trees in parallel
        Parallel(n_jobs=n_jobs, verbose=verbose, require='sharedmem')(
                delayed(compute_feature_contributions_from_tree)(
                        estimator=estimator, i_tree=i_tree,
                        data=self.data, targets=self.targets,
                        paths=self.tree_path[i_tree], features_split=None,
                        compute_marginal_contributions=True,
                        compute_conditional_contributions=False,
                        results=results, lock=threading.Lock())
                for i_tree, estimator in enumerate(getattr(self.model, self._internals['estimators_'])))

        if np.any(results['contributions_n_evaluations'] == 0):
            n_not_evaluated_features = np.unique(np.where(results['contributions_n_evaluations'] == 0)[1]).shape[0]
            warn('%i out %i (%.1f%%) features were never evaluated by the model.\nConsider increasing the number of estimators' % (
                    n_not_evaluated_features, results['contributions_n_evaluations'].shape[1], n_not_evaluated_features / results['contributions_n_evaluations'].shape[1] * 100))

        # Divide contributions only by the number of times the feature was evaluated.
        # Features that were never evaluated will return NaN
        self.contributions = divide0(results['contributions'], results['contributions_n_evaluations'],
                                     replace_with=np.nan)
        self.data_leaves = results['data_leaves']

        if verbose:
            print('done')

        return self


    def explain_interactions(self, data, targets=None, n_jobs=-1, verbose=False):
        """Main method to explain conditional contributions (that is,
        interactions) between features in the data.

        REFERENCE: This function is loosely based on _predict_forest in the python
        package treeinterpreter.
        SOURCE: https://github.com/andosa/treeinterpreter/blob/master/treeinterpreter/treeinterpreter.py

        :param data: [numpy array or pandas DataFrame] Data on which to test
            feature contributions. It must have the same number of features of
            the dataset used to train the model. It should be scaled in the same
            way as the training data.
        :param targets: [numpy array] True target values of each instance in
            `data`. For classification tasks, it contains the class labels, for
            regression problems the true value.
        :param n_jobs: [int or None] The number of parallel processes to use if
            joblib is installed. If None, trees are processed sequentially.
        :param verbose: [bool] Whether to print messages to the console regarding
            progress and outcomes.

        :return self: This allows the user to call this method together with
            initialization, and return the object in a variable, that is,
            TE = TreeExplainer(model).explain_interactions(X_test)

        The following attributes are stored in self:
        conditional_contributions: [dict] A dictionary containing the values of
            contribution of each feature conditioned on previous features. Each
            key contains the list of features along the decision path, from the
            root down to the leaf. Each dictionary value contains a numpy array
            containing the conditional contribution of each feature to each
            observation, averaged across trees.
        """

        # Preprocess and store data
        self._process_and_predict(data=data, targets=targets, calling_method='explain_interactions')

        if verbose:
            print('Computing conditional feature contributions ...')

        # Compute contributions
        results = dict()
        results['data_leaves'] = np.zeros((self.n_samples, self.n_trees), dtype=int)
        no_of_nodes_per_tree = [i.shape[0] for i in self.features_split]
        results['conditional_contributions'] = list([np.zeros((i, self.n_target_levels)) for i in no_of_nodes_per_tree])
        results['conditional_contributions_sample'] = list(np.empty((self.n_trees, ), dtype=object))
        # Process trees in parallel
        Parallel(n_jobs=n_jobs, verbose=verbose, require='sharedmem')(
                delayed(compute_feature_contributions_from_tree)(
                        estimator=estimator, i_tree=i_tree,
                        data=self.data, targets=self.targets,
                        paths=self.tree_path[i_tree],
                        features_split=self.features_split[i_tree],
                        compute_marginal_contributions=False,
                        compute_conditional_contributions=True,
                        results=results, lock=threading.Lock())
                for i_tree, estimator in enumerate(getattr(self.model, self._internals['estimators_'])))

        # Store conditional contributions
        self.conditional_contributions = results['conditional_contributions']
        self.conditional_contributions_sample = results['conditional_contributions_sample']
        self.data_leaves = results['data_leaves']

        if verbose:
            print('done')

        return self


    def explain_single_prediction(self, observation_idx, solve_duplicate_splits='mean',
                                  threshold_contribution=None, top_n_features=None,
                                  n_jobs=-1):
        """Analyze tree structure and try to explain how the model has reached a
        certain prediction for a single observation. The idea is to look at how
        each tree has used data features to partition the feature space, and
        how that rule generalizes across trees.

        :param observation_idx: [int] The index of an observation in the stored
            data
        :param solve_duplicate_splits: [str] Not implemented yet.
        :param threshold_contribution: [None or int] The threshold on
            contribution values below which features will be hidden from the final
            summary because uninformative. If None, nothing happens.
        :param top_n_features: [int or None] The number of most informative
            features, as measured by conditional contributions. If None,
            nothing happens.
        :param n_jobs: [int or None] The number of parallel processes to use if
            joblib is installed. If None, trees are processed sequentially.

        :return [str]: Prints message to console regarding the contribution of
            features to a single prediction.
        """

        if self.data is None:
            raise ValueError('No data is present. First run the method explain_interactions()')

        # Get data of observation
        this_sample_original = self.data[observation_idx, :]
        this_sample = [str(i).rstrip('0') for i in this_sample_original]
        # Convert data of this sample
        for i_feature, feature in enumerate(self.feature_names):
            if self.features_data_types[feature]['data_type'] == 'numerical':
                continue
            else:
                this_sample[i_feature] = self.features_data_types[feature]['categories'][int(this_sample_original[i_feature])]

        # Gather unique values from each feature
        feature_values = dict({feature: list() for feature in self.feature_names})
        for i_feature, feature in enumerate(self.feature_names):
            feature_values[feature] = np.unique(self.data[:, i_feature])

        # Process trees in parallel
        results = dict()
        results['samples_for_decision_table'] = dict({i: list() for i in self.feature_names})
        results['contributions_for_decision_table'] = dict({i: list() for i in self.feature_names})

        Parallel(n_jobs=n_jobs, verbose=False, require='sharedmem')(
                delayed(compute_explanation_of_prediction)(
                        leaf=self.data_leaves[observation_idx, i_tree],
                        paths=self.tree_path[i_tree],
                        threshold_value=self.threshold_value[i_tree],
                        features_split=self.features_split[i_tree],
                        conditional_contributions=self.conditional_contributions[i_tree],
                        prediction=self.predictions[observation_idx],
                        feature_values=feature_values,
                        solve_duplicate_splits=solve_duplicate_splits,
                        results=results,
                        lock=threading.Lock())
                for i_tree in range(self.n_trees))

        # Extract data
        samples_for_decision_table = results['samples_for_decision_table']
        contributions_for_decision_table = results['contributions_for_decision_table']

        # Initialize output variables
        numerical_columns = ['lower quartile', 'median', 'upper quartile']
        categorical_columns = ['1st choice', '2nd choice']
        # Split numerical from categorical features
        numerical_features = [feature for feature in self.feature_names
                              if self.features_data_types[feature]['data_type'] == 'numerical']
        categorical_features = list(set(self.feature_names).difference(numerical_features))
        # Make DataFrames
        decision_table_numerical = pd.DataFrame(columns=['value'] + numerical_columns + ['contribution'],
                                      index=numerical_features)
        decision_table_categorical = pd.DataFrame(columns=['value'] + categorical_columns + ['contribution'],
                                                  index=categorical_features)
        # Create function for filling the categorical DataFrame
        fill_categorical_values = lambda x, y: '%s (%i%%)' % (x, y * 100) if x != '' else ''
        fill_contribution_values = lambda x: '%.1f%%' % x

        # Compute summary statistics for each feature
        for feature in self.feature_names:
            samples = np.array(samples_for_decision_table[feature])

            # Check data type
            if feature in categorical_features:
                # Convert indices to values
                samples_value = self.features_data_types[feature]['categories'][samples.astype(int)]
                category_frequencies = pd.value_counts(samples_value, ascending=False, normalize=True)
                # Take the top 3 choices
                choices = list(category_frequencies.index)
                first_choice = choices[0]
                second_choice = choices[1] if len(choices) > 1 else ''
                # third_choice = choices[2] if len(choices) > 2 else ''

                # Take their frequency value
                category_frequencies = category_frequencies.values
                if category_frequencies.shape[0] < 2:
                    category_frequencies = np.hstack((category_frequencies,
                                                      np.zeros((2 - category_frequencies.shape[0], ))))

                # Store values in nice format
                decision_table_categorical.loc[feature, '1st choice'] = fill_categorical_values(first_choice, category_frequencies[0])
                decision_table_categorical.loc[feature, '2nd choice'] = fill_categorical_values(second_choice, category_frequencies[1])
                # decision_table_categorical.loc[feature, '3rd choice'] = fill_categorical_values(third_choice, category_frequencies[2])

                # Store median contribution
                decision_table_categorical.loc[feature, 'contribution'] = np.median(contributions_for_decision_table[feature])

            elif feature in numerical_features:
                # Compute quartiles
                q = np.quantile(samples, [.25, .50, .75], interpolation='nearest')
                decision_table_numerical.loc[feature, ['lower quartile', 'median', 'upper quartile']] = q

                # Store median contribution
                decision_table_numerical.loc[feature, 'contribution'] = np.median(contributions_for_decision_table[feature])

        # Add sample of interest to decision table
        decision_table_numerical['value'] = [this_sample[self.feature_names.index(i)] for i in numerical_features]
        decision_table_categorical['value'] = [this_sample[self.feature_names.index(i)] for i in categorical_features]

        # Sort decision table by contribution value
        decision_table_numerical.sort_values(by='contribution', ascending=False, inplace=True)
        decision_table_categorical.sort_values(by='contribution', ascending=False, inplace=True)

        # Limit number of features used to explain this prediction
        if threshold_contribution is not None:
            decision_table_numerical = decision_table_numerical.loc[decision_table_numerical['contribution'] >= threshold_contribution]
            decision_table_categorical = decision_table_categorical.loc[decision_table_categorical['contribution'] >= threshold_contribution]

        if top_n_features is not None:
            decision_table_numerical = decision_table_numerical.iloc[:top_n_features]
            decision_table_categorical = decision_table_categorical.iloc[:top_n_features]

        # Convert contribution column to string
        decision_table_numerical['contribution'] = decision_table_numerical['contribution'].map(fill_contribution_values)
        decision_table_categorical['contribution'] = decision_table_categorical['contribution'].map(fill_contribution_values)

        # Print to console
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            outcome = self.target_data_type[self.target_name]['categories'][int(self.predictions[observation_idx])]
            is_correct = 'correct' if self.correct_predictions[observation_idx] else 'not correct'
            print('\nObservation #%i: %s = \'%s\' (%s)\n' % (observation_idx, self.target_name, outcome, is_correct))
            print(decision_table_numerical)
            print()
            print(decision_table_categorical)
            print()


    def _process_and_predict(self, data, targets=None, calling_method=None):
        # Process input data
        DP = DataProcessor().prepare(data=data, targets=targets)

        # Check what to do
        did_have_data = self.data is not None
        did_get_data = DP.data is not None

        if did_get_data:
            if did_have_data:
                if np.array_equal(self.data, DP.data):
                    return  # Passed the same data
                else:
                    pass  # Passed new data, which will overwrite previous data
            else:
                pass  # Passed new data, which will fill attributes `X` and `y`
        else:
            if did_have_data:
                return  # No new data has been provided and there is data already
            else:
                raise ValueError('Please provide data and/or targets to the method %s()' % calling_method)

        # Extract info
        self.data = DP.data
        self.n_samples = self.data.shape[0]
        self.targets = DP.targets
        self.original_feature_names = DP.info['original_feature_names']
        self.feature_names = DP.info['feature_names']
        self.n_features = DP.info['n_features']
        self.target_levels = DP.info['target_levels']
        self.target_name = DP.info['target_name']
        self.n_target_levels = DP.info['n_target_levels']
        self.features_data_types = DP.info['features_data_types']
        self.target_data_type = DP.info['target_data_type']

        # Initialize other attributes that depend on data
        self.n_samples = None
        self.predictions = None
        self.correct_predictions = None
        self.contributions = None
        self.conditional_contributions = None
        self.feature_combinations = None
        self.conditional_contributions_sample = None
        self.data_leaves = None
        self.min_depth_frame = None
        self.min_depth_frame_summary = None
        self.importance_frame = None
        self.two_way_contribution_table = None
        self.n_two_way_contribution_table = None

        # Compute and store predictions
        self.prediction_probabilities = self.model.predict_proba(self.data)
        if self._internals['model_type'] == 'classifier':
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)
            if self.targets is not None:
                self.correct_predictions = self.predictions == self.targets
        else:
            self.predictions = self.model.predict(self.data)

        # Update feature names if they changed
        if list(self.feature_depth.keys()) != self.feature_names:
            self.feature_depth = dict({self.feature_names[idx]: value for idx, (_, value) in enumerate(self.feature_depth.items())})
        if list(self.no_of_nodes.keys()) != self.feature_names:
            self.no_of_nodes = dict({self.feature_names[idx]: value for idx, (_, value) in enumerate(self.no_of_nodes.items())})


    ############################################################################
    # Statistics
    ############################################################################
    def compute_min_depth_distribution(self, mean_sample='relevant_trees'):
        """Calculates distribution of minimal depth of all variables in all trees.

        REFERENCE: This function is based on plot_min_depth_distribution in the
        R package randomForestExplainer.
        SOURCE: https://github.com/MI2DataLab/randomForestExplainer/blob/master/R/min_depth_distribution.R

        :param mean_sample:
            - If mean_sample = "all_trees" (filling missing value): the minimal
            depth of a variable in a tree that does not use it for splitting is
            equal to the mean depth of trees. Note that the depth of a tree is
            equal to the length of the longest path from root to leave in this
            tree. This equals the maximum depth of a variable in this tree plus
            one, as leaves are by definition not split by any variable.
            - If mean_sample = "top_trees" (restricting the sample): to
            calculate the mean minimal depth only B^tilde out of B (number of
            trees) observations are considered, where B^tilde is equal to the
            maximum number of trees in which any variable was used for
            splitting. Remaining missing values for variables that were used for
            splitting less than B^tilde times are filled in as in mean_sample =
            "all_trees".
            - mean_sample = "relevant_trees" (ignoring missing values): mean
            minimal depth is calculated using only non-missing values.

        The following attributes are stored in self:
        min_depth_frame: [pandas DataFrame] Contains the depth at which each
            feature can be found in each tree.
        min_depth_frame_summary: [pandas DataFrame] Contains the count of each
            depth value for each feature.
        """

        # Check inputs
        if mean_sample != 'relevant_trees':
            raise NotImplementedError

        # Initialize temporary variables
        new_depth_value = None

        # Convert data to a long format
        feature = np.vstack([np.tile(key, (len(value), 1))
                             for key, value in self.feature_depth.items()])
        depth_value = np.hstack([np.array(value) for value in
                                 self.feature_depth.values()])
        min_depth_frame = pd.DataFrame(columns=['tree', 'feature', 'minimal_depth'])
        min_depth_frame['tree'] = np.tile(np.arange(self.n_trees) + 1, (1, self.n_features)).ravel()
        min_depth_frame['minimal_depth'] = depth_value
        # Features become a categorical data type
        min_depth_frame['feature'] = pd.Categorical(feature.ravel(),
                                                    categories=self.feature_names,
                                                    ordered=True)
        # Sort output as in randomForestExplainer
        min_depth_frame.sort_values(by=['tree', 'feature'], ascending=True, inplace=True)
        min_depth_frame.reset_index(drop=True, inplace=True)
        # Drop rows where minimal_depth is negative because it means that the
        # feature was not used by that tree
        min_depth_frame.drop(np.where(min_depth_frame['minimal_depth'] < 0)[0], inplace=True)
        min_depth_frame.reset_index(drop=True, inplace=True)

        # Summarize data by reporting count of each [feature minimal_depth] combination
        summary = min_depth_frame.groupby(['feature', 'minimal_depth']).size()
        # Convert back to DataFrame
        summary = summary.to_frame(name='count').reset_index(level=['feature', 'minimal_depth'])
        # Impute depth of features for those that were not evaluated in all trees
        if mean_sample != 'relevant_trees':
            missing_values = summary.groupby('feature').sum()
            missing_values['n_missing_trees'] = self.n_trees - missing_values['count']
            missing_values = missing_values[missing_values['n_missing_trees'] > 0]
            if missing_values.shape[0] > 0:
                rows_to_add = list()
                features_with_missing_values = list(missing_values.index)
                for feature in features_with_missing_values:
                    if mean_sample == 'all_trees':
                        new_depth_value = self.tree_depth.mean()
                    elif mean_sample == 'top_trees':
                        raise NotImplementedError
                    # Store values
                    rows_to_add.append([feature, new_depth_value,
                                        missing_values.loc[feature]['n_missing_trees']])
                # Add missing values to summary data
                summary = summary.append(pd.DataFrame(rows_to_add, columns=summary.columns), ignore_index=True)
                summary.sort_values(by=['feature', 'minimal_depth'], ascending=True, inplace=True)
                summary.reset_index(drop=True, inplace=True)

        # Store outputs
        self.min_depth_frame = min_depth_frame
        self.min_depth_frame_summary = summary


    def compute_two_way_interactions(self, n_jobs=-1, verbose=False):
        """This function computes tables of two-way conditional interactions
        between features. These values are the relative change in feature
        contribution at a node where a feature is in the parent node, and another
        one in one of the children. This type of information could highlight
        whether there are combinations of features that are used in sequence
        more often than others. The last column of the table reports the
        relative contribution of a feature when used at the root of the tree.

        Values are averaged across trees and observations in the data provided
        to the explain_features() method.

        If the model is a classifier, we can further  divide feature
        contributions between correct and incorrect predictions of the model,
        to characterize how features interact when the model turns out to be
        right and when it is wrong.

        :param n_jobs: [int or None] The number of parallel process to use.
        :param verbose: [bool] Whether to print progress messages.

        The following attributes are stored in self:
        two_way_contribution_table: [dict] Contains pandas DataFrames of
            relative feature contributions when 'all' samples are used. If the
            model is a classifier, also tables for 'correct' and 'incorrect'
            predictions will be calculated. In each table, larger values indicate
            that when a particular feature is used after another one, there is a
            more pronounced change in the prediction.
        n_two_way_contribution_table: [dict] Same as two_way_contribution_table.
            However, it contains the number of times a feature was used for split.
            Higher values stand for a more frequent use of a combination of
            features, indicating a possible relationship between them.
        """
        # Compute conditional contributions if not previously done
        if self.conditional_contributions is None:
            self.explain_interactions(data=None, targets=None,
                                      n_jobs=n_jobs, verbose=verbose)

        # Initialize temporary variables
        key_name = None
        samples = None

        # If the model is a classifier, separate contributions for correct and
        # incorrect predictions of the model
        if self._internals['model_type'] == 'classifier':
            n_iterations = 3
        else:
            n_iterations = 1

        # Initialize output variables
        self.two_way_contribution_table = dict()
        self.n_two_way_contribution_table = dict()
        for i_iter in range(n_iterations):
            if i_iter == 0:  # Select all samples
                samples = np.arange(self.n_samples)
                key_name = 'all'

            else:
                if self._internals['model_type'] == 'classifier':
                    if i_iter == 1:  # Select only correctly predictions samples
                        samples = np.where(self.targets == self.predictions)[0]
                        key_name = 'correct'
                    elif i_iter == 2:  # Select only wrongly predicted samples
                        samples = np.where(self.targets != self.predictions)[0]
                        key_name = 'incorrect'

            if verbose:
                print('Computing two-way feature interactions of %s predictions ...' % key_name)

            # Initialize heatmap-tables
            results = dict()
            results['contribution_values'] = np.zeros((self.n_features,
                                                       self.n_features + 1,
                                                       self.n_target_levels),
                                                      dtype=np.float32)
            results['contribution_values_n'] = np.zeros((self.n_features,
                                                         self.n_features + 1),
                                                        dtype=int)

            # Process trees in parallel
            Parallel(n_jobs=n_jobs, verbose=verbose, require='sharedmem')(
                    delayed(compute_two_way_conditional_contributions)(samples,
                                                                       self.conditional_contributions[i_tree],
                                                                       self.conditional_contributions_sample[i_tree],
                                                                       self.features_split[i_tree],
                                                                       results, threading.Lock())
                    for i_tree, estimator in enumerate(getattr(self.model, self._internals['estimators_'])))

            # Store values
            # Average interactions across all samples and trees. Combinations of
            # features that have never been selected will get a NaN
            self.two_way_contribution_table[key_name] = divide0(results['contribution_values'],
                                                                np.atleast_3d(results['contribution_values_n']),
                                                                replace_with=np.nan)
            self.n_two_way_contribution_table[key_name] = results['contribution_values_n']

        if verbose:
            print('done')


    ################################################################################
    # Summary
    ################################################################################
    def summarize_importance(self, permutation_iterations=0, display=True):
        """Calculate different measures of importance for variables presented
        in the model. Different variables are available for classification
        and regression models.

        REFERENCE: This function is based on the function measure_importance in
        the R package randomForestExplainer.
        SOURCE: https://github.com/MI2DataLab/randomForestExplainer/blob/master/R/measure_importance.R

        :param permutation_iterations: [int > 0] The number of permutations to
            compute the 'significance' of the importance value of each feature.
            If 0, the permutation test is skipped.
        :param display: [bool] Whether to display the results in the console.

        :return importance_frame: [pandas DataFrame] Contains importance metrics
            for the model.
        """

        # Initialize temporary variables
        node_purity = None

        # Compute the minimal depth distribution, if not done already
        if self.min_depth_frame is None:
            self.compute_min_depth_distribution()

        # Initialize importance_frame
        if self._internals['model_type'] == 'classifier':
            accuracy_column_name = 'accuracy_decrease'
            node_purity_column_name = 'gini_decrease'
        else:
            accuracy_column_name = 'mse_increase'
            node_purity_column_name = 'node_purity_increase'
        importance_frame = pd.DataFrame(columns=['variable', 'mean_min_depth',
                                                 'no_of_nodes', accuracy_column_name,
                                                 node_purity_column_name,
                                                 'no_of_trees', 'times_a_root',
                                                 'p_value'])
        for i_feature, feature in enumerate(self.feature_names):
            # Gather info on this feature from other tables
            mean_min_depth = self.min_depth_frame[self.min_depth_frame['feature'] == feature]['minimal_depth'].mean()
            no_of_nodes = self.no_of_nodes[feature].sum()
            min_depth_summary_this_feature = self.min_depth_frame_summary[self.min_depth_frame_summary['feature'] == feature]
            no_of_trees = min_depth_summary_this_feature['count'].sum()
            if (min_depth_summary_this_feature['minimal_depth'] == 0).any():
                times_a_root = min_depth_summary_this_feature[min_depth_summary_this_feature['minimal_depth'] == 0]['count'].values[0]
            else:
                times_a_root = 0

            # Compute performance information based on the model type
            if permutation_iterations > 0:
                raise NotImplementedError
                # if accuracy_column_name == 'accuracy_decrease':
                #     accuracy = 1
                # elif accuracy_column_name == 'mse_increase':
                #     accuracy = 1
            else:
                accuracy = np.nan

            if node_purity_column_name == 'gini_decrease':
                node_purity = np.nan
            elif node_purity_column_name == 'node_purity_increase':
                node_purity = np.nan

            # Compute p-value
            p_value = binom_test(no_of_nodes, no_of_nodes, 1 / self.n_features,
                                 alternative='greater')

            # Store data
            importance_frame.at[i_feature, 'variable'] = feature
            importance_frame.at[i_feature, 'mean_min_depth'] = mean_min_depth
            importance_frame.at[i_feature, 'no_of_nodes'] = no_of_nodes
            importance_frame.at[i_feature, accuracy_column_name] = accuracy
            importance_frame.at[i_feature, node_purity_column_name] = node_purity
            importance_frame.at[i_feature, 'no_of_trees'] = no_of_trees
            importance_frame.at[i_feature, 'times_a_root'] = times_a_root
            importance_frame.at[i_feature, 'p_value'] = p_value

        # Remove the accuracy column, if that metric has not been computed
        if permutation_iterations == 0:
            importance_frame.drop(columns=accuracy_column_name, inplace=True)

        # Sort values
        sort_by = accuracy_column_name if permutation_iterations > 0 else node_purity_column_name
        importance_frame.sort_values(by=sort_by, ascending=False, inplace=True)
        importance_frame.reset_index(drop=True, inplace=True)

        # Store results
        self.importance_frame = importance_frame

        # Display results
        if display:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(importance_frame)


    ############################################################################
    # Plots
    ############################################################################
    def plot_min_depth_distribution(self, top_n_features=None,
                                    min_trees_fraction=0.0, mark_average=True,
                                    average_n_digits=2, sort_by_weighted_mean=False,
                                    title='Distribution of minimal depth',
                                    colormap='tab20', return_figure_handle=False):
        """Plots distribution of minimal depth of variables in all trees along
        with mean depths for each variable. In general, the shallower (less deep)
        variables are the more influential.

        REFERENCE: This function has been inspired by plot_min_depth_distribution
        in the R package randomForestExplainer.
        SOURCE: https://github.com/MI2DataLab/randomForestExplainer/blob/master/R/min_depth_distribution.R

        :param top_n_features: [int or None] The maximal number of variables with
            lowest mean minimal depth to plot. If None, all features are shown.
        :param min_trees_fraction: [float in range [0, 1], extrema included] The
            fraction of trees in which a feature has to be used for splitting
            to have the feature included in the plot.
        :param mark_average: [bool] Whether to mark the average depth on the plot.
        :param average_n_digits: [int] Number of digits for displaying mean
            minimal depth.
        :param sort_by_weighted_mean: [bool] Whether to sort features by their
            proportion in each depth bin. In thi way, features that appeared more
            often at a shorted depth will rank higher, despite their actual mean.
        :param title: [str] The plot title.
        :param colormap: [str] Name of matplotlib colormap. Default is 'tab20'.
        :param return_figure_handle: [bool] Whether to return the figure handle,
            which can be used for printing, for example.

        :return fig: [matplotlib Figure] The displayed figure.
        """

        # Cannot continue with minimal depth distributions
        if self.min_depth_frame is None:
            raise ValueError('Needs to compute minimal depth distributions first.\nUse method compute_min_depth_distribution() to do so.')

        # Get number of trees in which a feature was used for splitting
        tree_count = self.min_depth_frame_summary.groupby('feature')['count'].sum().to_frame()
        tree_count['fraction'] = tree_count['count'] / self.n_trees
        tree_count = tree_count[tree_count['fraction'] >= float(min_trees_fraction)]
        # Get list of features to analyze
        features = tree_count.index.to_list()

        # Get the max depth among all trees
        max_depth = max([value.max() for key, value in self.feature_depth.items()
                         if key in features])
        # Make colormap
        cmap = plt.get_cmap(colormap, max_depth)

        # Compute average minimum depth and the position of each label
        avg_min_depth = pd.DataFrame(columns=['feature', 'avg_depth', 'x', 'weight'])
        for i_feature, feature in enumerate(features):
            data = self.min_depth_frame_summary.loc[self.min_depth_frame_summary['feature'] == feature,
                                                    ['minimal_depth', 'count']]
            # If user did not request it, do not calculate the average
            if mark_average:
                this_feature_depth_values = self.min_depth_frame[
                    self.min_depth_frame['feature'] == feature]['minimal_depth']
                avg_depth = this_feature_depth_values.mean()
                sorted_depth_values = np.hstack([np.linspace(data.iloc[i]['minimal_depth'],
                                                             data.iloc[i]['minimal_depth'] + 1,
                                                             data.iloc[i]['count'])
                                                 for i in range(data.shape[0])])
                mean_depth_pos = np.abs(sorted_depth_values - avg_depth).argmin()
                mean_depth_pos = np.clip(mean_depth_pos, a_min=0, a_max=self.n_trees).astype(int)
            else:
                avg_depth = np.nan
                mean_depth_pos = np.nan

            # Store values
            avg_min_depth.at[i_feature, 'feature'] = feature
            avg_min_depth.at[i_feature, 'avg_depth'] = avg_depth
            avg_min_depth.at[i_feature, 'x'] = mean_depth_pos
            avg_min_depth.at[i_feature, 'weight'] = (data['count'] * data['minimal_depth'] ** 2).sum()

        # Sort values
        if sort_by_weighted_mean:
            # Features used closer to the root more often will rank higher
            sort_by = 'weight'
        else:
            sort_by = 'avg_depth'
        # Apply sorting
        avg_min_depth.sort_values(sort_by, ascending=True, inplace=True)
        avg_min_depth.reset_index(drop=True, inplace=True)
        # Re-extract (sorted) list of features
        features = avg_min_depth['feature'].tolist()
        # Keep only top features
        if top_n_features is not None:
            features = features[:top_n_features]
        # Generate a color for each depth level
        depth_values = np.arange(max_depth + 1)
        # Get location and width of each bar
        n_features = len(features)
        feature_y_width = 1 / n_features * .9
        feature_y_pos = np.linspace(0, 1, n_features)
        feature_y_gap = feature_y_pos[1] - feature_y_width

        # Open figure
        fig = plt.figure(figsize=(7, 8))
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        # Mark the maximum number of trees used
        max_n_trees = 0
        # Plot horizontally stacked bars
        for i_feature, feature in enumerate(features):
            # Get data and prepare x- and y-ranges
            data = self.min_depth_frame_summary.loc[self.min_depth_frame_summary['feature'] == feature,
                                                    ['minimal_depth', 'count']]
            # Add missing depths
            missing_depths = np.setdiff1d(depth_values, data['minimal_depth'])
            data = pd.concat((data, pd.DataFrame(np.vstack((missing_depths, np.zeros(missing_depths.shape[0]))).T.astype(int),
                                                 columns=data.columns)), ignore_index=True,
                             sort=True)
            data.sort_values(by='minimal_depth', ascending=True, inplace=True)
            # Get count
            count = data.sort_values(by='minimal_depth')['count'].values
            count = np.vstack((np.cumsum(count) - count, count)).T
            max_n_trees = max(max_n_trees, count.max())
            # Plot horizontal bars
            yrange = (feature_y_pos[i_feature], feature_y_width)
            ax.broken_barh(xranges=count.tolist(), yrange=yrange, facecolors=cmap.colors,
                           alpha=.8)
            # Mark average depth
            if mark_average is not None:
                # Add vertical bar for the mean
                ax.plot([avg_min_depth.loc[i_feature, 'x']] * 2, [yrange[0], yrange[0] + yrange[1]],
                        color='k', lw=5, solid_capstyle='butt')
                # Add text box showing the value of the mean
                ax.text(avg_min_depth.loc[i_feature, 'x'], yrange[0] + yrange[1] / 2,
                        '%%.%if' % average_n_digits % avg_min_depth.loc[i_feature, 'avg_depth'],
                        ha='center', va='center', bbox=dict(boxstyle='round', facecolor='w'))

        # Adjust axes appearance
        ax.set_yticks(feature_y_pos + feature_y_width / 2)
        ax.set_yticklabels(features)
        ax.set_ylim(1 + feature_y_width + feature_y_gap, 0)
        ax.set_xlim(0, max_n_trees)
        adjust_spines(ax, spines=['bottom', 'left'], offset=0, smart_bounds=True)
        ax.spines['left'].set_color('None')
        ax.tick_params(axis='y', length=0, pad=0)
        ax.set_xlabel('Number of trees (out of %i)' % self.n_trees)
        if top_n_features is not None:
            title += ' (top %i features)' % top_n_features
        ax.set_title(title)
        # Adjust layout
        fig.tight_layout()
        # Add lines at beginning and end of plotting area
        ax.axvline(0, color='k', lw=.5)
        ax.axvline(self.n_trees, color='k', lw=.5)

        # Add colorbar
        cmap_cbar = LinearSegmentedColormap.from_list('cmap', cmap.colors, cmap.N + 1)
        ax_bg = fig.add_axes(ax.get_position())
        im_cbar = ax_bg.imshow(np.tile(depth_values, (2, 1)),
                               cmap=cmap_cbar, aspect='auto', interpolation=None,
                               vmin=depth_values.min(), vmax=depth_values.max() + 1)
        cbar = fig.colorbar(im_cbar)
        # The axes ax_bg has now been squeezed to the left by the colorbar. Copy
        # that position back to ax, and hide the axes ax_bg
        ax.set_position(ax_bg.get_position())
        ax_bg.set_visible(False)
        # Set labels of colorbar
        cbar.ax.tick_params(axis='both', length=0)
        cbar.set_ticks(depth_values + .5)
        cbar.set_ticklabels(depth_values)
        # Invert colorbar direction so 'deeper' is below 'shallower'
        cbar.ax.set_ylim(cbar.ax.get_ylim()[::-1])
        # Make colorbar shorter so a title can be written
        bbox = cbar.ax.get_position()
        cbar.ax.set_position([bbox.x0 + .025, bbox.y0, bbox.x1 - bbox.x0, .6])
        cbar.ax.set_title('Minimal\ndepth')

        if return_figure_handle:
            return fig


    def plot_two_way_interactions(self, sort_features_on_target=True,
                                  targets_to_plot=None, sort_on_contributions=True,
                                  top_n_features=None, return_fig_handle=False):

        # Cannot continue without 2way tables
        if self.two_way_contribution_table is None:
            raise ValueError('Needs to compute 2-way interactions between features first.\nUse method compute_two_way_interactions() to do so.')

        # If sort_features_on_target is True, and which target is unspecified,
        # use the first target that will be plotted
        if isinstance(sort_features_on_target, bool):
            if sort_features_on_target:
                sort_features_on_target = targets_to_plot[0]

        # Set number of subplot columns in figure
        if targets_to_plot is not None:
            targets_to_plot = true_list(targets_to_plot)

        else:
            if isinstance(sort_features_on_target, str):
                targets_to_plot = list([sort_features_on_target])

            elif self._internals['model_type'] == 'classifier' and self.n_target_levels == 2:
                # If this is a binary classification task, the contribution to each
                # target will be identical, so we'll plot only one
                targets_to_plot = list([self.target_levels[-1]])  # Assume that more interesting label is the highest

            else:
                targets_to_plot = list(self.target_levels)

        # Get indices of targets to plot
        targets_to_plot_idx = [self.target_levels.index(i) for i in targets_to_plot]
        n_columns = len(targets_to_plot_idx)
        # Add another plot for the heatmap with percentage of values
        n_columns += 1

        # A column for each heatmap
        tables = list(self.two_way_contribution_table.keys())
        n_figures = len(tables)
        figures = list()

        for i_fig in range(n_figures):
            # If sort_features_on_target is a string, it is the name of the target
            # to use. In that case, get a sorting order for the columns
            if isinstance(sort_features_on_target, str):
                # Select corresponding data
                index_of_target = self.target_levels.index(sort_features_on_target)
                if sort_on_contributions:
                    data = self.two_way_contribution_table[tables[i_fig]][:, :, index_of_target]
                else:
                    data = self.n_two_way_contribution_table[tables[i_fig]]
                # Convert to DataFrame
                df = pd.DataFrame(data, index=self.feature_names,
                                  columns=self.feature_names + ['root'])
                # Sort by contribution when feature is used at the root
                df['root'].replace(to_replace=np.nan, value=0, inplace=True)
                df.sort_values(by='root', ascending=False, inplace=True,
                               na_position='last')
                # Get order of features
                features_order = list(df.index)

            else:
                features_order = list(self.feature_names)

            # Keep only at most n features
            if top_n_features is not None:
                features_order = features_order[:top_n_features]

            # Open figure and plot heatmaps
            fig, ax = plt.subplots(nrows=1, ncols=n_columns, figsize=(13, 5))
            figures.append(fig)
            for i_col in range(n_columns):
                # Assess whether to add an annotation to the heatmap
                heatmap_annotations = len(features_order) < 15

                # Select data
                if i_col < n_columns - 1:  # Contribution values go in the first n-1 plots
                    data = self.two_way_contribution_table[tables[i_fig]][:, :, targets_to_plot_idx[i_col]]
                    subplot_title = 'Predictions for \'%s\'' % self.target_levels[targets_to_plot_idx[i_col]]

                else:  # Percentages go in the last plot
                    data = self.n_two_way_contribution_table[tables[i_fig]]
                    # Normalize across number of trees and samples to
                    # obtain a percentage
                    data = data / self.n_trees / self.n_samples * 100
                    subplot_title = 'Combination frequencies'

                # Remove 0s
                data[data == 0] = np.nan
                # Convert data to DataFrame, so it can be passed to seaborn
                df = pd.DataFrame(data, index=self.feature_names,
                                  columns=self.feature_names + ['root'])
                # Move root to the front
                df = pd.concat((df['root'], df[self.feature_names]), axis=1)

                # Sort features
                if features_order is not None:
                    # Select and sort columns
                    df = pd.concat((df['root'], df[features_order]), axis=1)
                    # Select and sort rows
                    df = df.loc[features_order]

                # Determine whether annotation should be displayed
                data = df.values.ravel()
                if np.all(np.abs(data[np.isfinite(data)]) < 1):
                    heatmap_annotations = False

                # Add column for sum contribution across all combinations
                # if i_col < n_columns - 1:  # Plot contribution values
                #     marginal_contribution = self.contributions[:, :, targets_to_plot[i_col]].mean(axis= 0)
                #     # Sort features, if necessary
                #     if features_order is not None:
                #         marginal_contribution = [marginal_contribution[features_order.index(i)] for i in self.feature_names]
                #     # Store values
                #     df['marginal'] = marginal_contribution

                # Plot heatmap
                ax[i_col] = sns.heatmap(df, cmap='RdBu_r', center=0,
                                        square=True, ax=ax[i_col], annot=heatmap_annotations,
                                        fmt='.0f')
                # Fix axes appearance
                ax[i_col].set_title(subplot_title)

            # Add title to figure
            fig.suptitle('%s%s predictions' % (tables[i_fig][0].upper(), tables[i_fig][1:]))
            plt.tight_layout()
            fig.subplots_adjust(top=.9, bottom=fig.subplotpars.bottom + .01)

        if return_fig_handle:
            return figures
