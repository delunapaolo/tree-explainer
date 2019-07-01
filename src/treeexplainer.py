# System
import threading
from joblib import Parallel, delayed
from warnings import warn

# Numerical
import numpy as np
import pandas as pd
from scipy.stats import binom_test

# Graphical
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Local repo
from core import validate_model_type, validate_model_is_trained
from core import _compute_tree_paths, _compute_feature_contributions_from_tree
from utilities.numerical import divide0
from utilities.visualization import adjust_spines


################################################################################
class TreeExplainer(object):
    def __init__(self, model, feature_names=None, target_names=None,
                 n_jobs=-1, verbose=False):
        """
        The class is instantiated by passing the model, and the names of features
        and targets in the data.

        :param model: The input model to explain.
        :param feature_names: [list] Name of each feature. Used for plots,
            output DataFrames and printing.
        :param target_names: [list] Name of each target. Used for plots, output
            DataFrames and printing. If model is a classifier, these are the
            names of the classes, if a regressor these are the names of the
            target variables.
        :param n_jobs: [int or None] The number of parallel processes to use.
        :param verbose: [bool] Whether to print progress to the console.
        """

        # Check that model type is supported
        model_class, model_type, implementation, estimator_type = validate_model_type(model)
        # Check that model has been fit
        n_estimators_attr, estimators_attr = validate_model_is_trained(model, model_class)

        # Store model and info on it
        self.model = model
        # Keep a reference to these attributes of the model
        self._internals = dict(model_class=model_class,
                               model_type=model_type,
                               implementation=implementation,
                               estimator_type=estimator_type,
                               estimators_=estimators_attr,
                               n_estimators=n_estimators_attr)
        # Initialize basic attributes of the model
        self.n_trees = getattr(self.model, self._internals['n_estimators'])
        self.n_features = self.model.n_features_
        if feature_names is None:
            feature_names = ['feature_%i' % (i + 1) for i in range(self.n_features)]
        self.feature_names = feature_names
        if target_names is None:
            if self._internals['model_type'] == 'classifier':
                target_names = ['target_%i' % (i + 1) for i in range(self.model.n_classes_)]
        self.target_names = target_names
        self.n_targets = len(self.target_names)

        # Extract decision path of each tree in model
        results = dict()
        results['tree_path'] = list(np.empty((self.n_trees, ), dtype=object))
        results['tree_depth'] = np.zeros((self.n_trees, ), dtype=np.int8)
        results['feature_depth'] = {f: np.zeros((self.n_trees, ), dtype=np.int8) - 1 for f in self.feature_names}
        results['value_threshold'] = {f: np.zeros((self.n_trees, ), dtype=np.float32) * np.nan for f in self.feature_names}
        results['no_of_nodes'] = {f: np.zeros((self.n_trees,), dtype=np.int8) - 1 for f in self.feature_names}
        Parallel(n_jobs=n_jobs, verbose=verbose, require='sharedmem')(
                delayed(_compute_tree_paths)(estimator, i_tree, results, threading.Lock())
                for i_tree, estimator in enumerate(getattr(self.model, self._internals['estimators_'])))
        # Store results
        self.tree_path = results['tree_path']
        self.tree_depth = results['tree_depth']
        self.feature_depth = results['feature_depth']
        self.value_threshold = results['value_threshold']
        self.no_of_nodes = results['no_of_nodes']
        # Initialize all other TreeExplainer's attributes
        self.predictions = None
        self.prediction_probabilities = None
        self.contributions = None
        self.conditional_contributions = None
        self.min_depth_frame = None
        self.min_depth_frame_summary = None
        self.importance_frame = None


    def explain(self, X_test, compute_conditional_contribution=False,
                n_jobs=-1, verbose=False):
        """Main method to explain the provided predictions of the model. It
        computes feature contributions, such that:
        predictions ≈ target_frequency_at_root + feature_contributions.

        REFERENCE: This function is based on _predict_forest in the python
        package treeinterpreter.
        SOURCE: https://github.com/andosa/treeinterpreter/blob/master/treeinterpreter/treeinterpreter.py

        :param X_test: [numpy or DataFrame] Data on which to test feature
            contributions. It must have the same number of features of the
            dataset used to train the model. It should be scaled in the same way
            as the training data.
        :param compute_conditional_contribution: [bool] Whether to also compute
            all conditional contributions along the path.
        :param n_jobs: [int or None] The number of parallel processes to use if
            joblib is installed. If None, trees are processed sequentially.
        :param verbose: [bool] Whether to print messages to the console regarding
            progress and outcomes.

        :return self: This allows the user to call this method together with
            initialization, and return the object in a variable, that is,
            TE = TreeExplainer(model).explain(X_test)

        The following attributes are stored in self:
        predictions: [numpy array] Contains the prediction of each feature to
            each observation and class, averaged across trees.
        target_frequency_at_root: [numpy array] Contains the baseline prediction
            of each feature to each observation and class, averaged across trees.
        contributions: [numpy array] Contains the contribution of each feature
            to each observation and class, averaged across trees.
        conditional_contributions: [dict] (optional if
            `compute_conditional_contribution` == True) A dictionary containing
            the values of contribution of each feature conditioned on previous
            features. Each key contains the list of features along the decision
            path, from the root down to the leaf. Each dictionary value contains
            a numpy array containing the conditional contribution of each
            feature to each observation, averaged across trees.
        """

        # Initialize TreeExplainer's attributes that depend on the supplied data
        self.predictions = None
        self.prediction_probabilities = None
        self.contributions = None
        self.conditional_contributions = None
        self.min_depth_frame = None
        self.min_depth_frame_summary = None
        self.importance_frame = None

        # Set feature names
        if isinstance(X_test, pd.DataFrame):
            # Import names of columns
            self.feature_names = X_test.columns
            did_change_feature_names = True
            # Convert X_test to numpy array
            X_test = X_test.values
        else:
            did_change_feature_names = False

        # Initialize output variable
        n_samples = X_test.shape[0]
        data_type = X_test.dtype
        # Compute and store predictions
        self.predictions = self.model.predict(X_test)
        self.prediction_probabilities = self.model.predict_proba(X_test)

        # Compute contributions
        results = dict()
        results['contributions'] = np.zeros((n_samples, self.n_features, self.n_targets), dtype=data_type)
        results['contributions_n_evaluations'] = np.zeros((n_samples, self.n_features, self.n_targets), dtype=data_type)
        results['conditional_contributions'] = dict()
        # results['conditional_contributions_samples'] = dict()
        # Process trees in parallel
        Parallel(n_jobs=n_jobs, verbose=verbose, require='sharedmem')(
                delayed(_compute_feature_contributions_from_tree)(
                        estimator, X_test, self.tree_path[i_tree],
                        compute_conditional_contribution, results, threading.Lock())
                for i_tree, estimator in enumerate(getattr(self.model, self._internals['estimators_'])))

        if compute_conditional_contribution:
            # Average across the same set of features used by multiple trees
            feature_sets = list(results['conditional_contributions'].keys())
            for features in feature_sets:
                if results['conditional_contributions'][features].ndim == 3:
                    results['conditional_contributions'][features] = \
                        np.mean(results['conditional_contributions'][features],
                                axis=2)

        if np.any(results['contributions_n_evaluations'] == 0):
            n_not_evaluated_features = np.unique(np.where(results['contributions_n_evaluations'] == 0)[1]).shape[0]
            warn('%i out %i (%.1f%%) features were never evaluated by the model.\nConsider increasing the number of estimators' % (
                    n_not_evaluated_features, results['contributions_n_evaluations'].shape[1], n_not_evaluated_features / results['contributions_n_evaluations'].shape[1] * 100))

        # Divide contributions only by the number of times the feature was evaluated.
        # Features that were never evaluated will return NaN
        self.contributions = divide0(results['contributions'], results['contributions_n_evaluations'],
                                     replace_with=np.nan)
        # We always save this variable to avoid that another call to this method would mix values referring to different X_test
        self.conditional_contributions = results['conditional_contributions']

        # Rename features
        if did_change_feature_names:
            raise NotImplementedError
            self.feature_depth = {self.feature_names[key]: value for key, value in
                                  self.feature_depth.items()}
            self.tree_depth = np.array(tree_depth)
            self.value_threshold = {self.feature_names[key]: value for key, value in
                                  self.value_threshold.items()}
            if compute_conditional_contribution:
                self.conditional_contributions = {self.feature_names[key]: value for key, value in
                                                  self.conditional_contributions.items()}

        return self


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
                node_purity = self.model.feature_importances_[i_feature]
            elif node_purity_column_name == 'node_purity_increase':
                raise NotImplementedError

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
                                    average_n_digits=2, sort_by_weighted_mean=True,
                                    title='Distribution of minimal depth',
                                    colormap='tab20'):
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

        :return fig: [matplotlib Figure] The displayed figure.
        """

        # Compute the minimal depth distribution, if not done already
        if self.min_depth_frame is None:
            self.compute_min_depth_distribution()

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
            count = np.vstack((np.cumsum(count) - count, count)).T.tolist()
            # Plot horizontal bars
            yrange = (feature_y_pos[i_feature], feature_y_width)
            ax.broken_barh(xranges=count, yrange=yrange, facecolors=cmap.colors,
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
        adjust_spines(ax, spines=['bottom', 'left'], offset=0, smart_bounds=True)
        ax.spines['left'].set_color('None')
        ax.tick_params(axis='y', length=0, pad=0)
        ax.set_xlabel('Number of trees')
        if top_n_features is not None:
            title += ' (top %i features)' % top_n_features
        ax.set_title(title)
        # Adjust layout
        fig.tight_layout()
        # Add lines at beginning and end of plotting area
        ax.axvline(0, color='k', lw=.5)
        ax.axvline(self.n_trees, color='k', lw=.5)

        # Add colorbar
        cmap_cbar = LinearSegmentedColormap.from_list('cmap', cmap.colors, cmap.N)
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

        return fig


    def plot_value_threshold(self, sort=True):
        raise NotImplementedError
        # Get summary data
        df = self.summarize_value_threshold(sort=sort, silent=True)

        # Open figure
        fig = plt.figure(figsize=(7, 8))
        fig.clf()
        # Make violin plot in gray
        ax = sns.violinplot(y='feature', x='value', data=df,
                            scale='count', orient='h', inner=None, color='.8',
                            width=.6, cut=0, bw=.2)
        # Overlay strip-plot
        ax = sns.stripplot(y='feature', x='value', data=df,
                           orient='h', palette='Set2', linewidth=1,
                           size=5, ax=ax, jitter=True)
        # Draw 0-line
        ax.axvline(0, color='r', lw=.5)
        # Adjust axes appearance
        adjust_spines(ax, ['bottom', 'left'], offset=3, smart_bounds=True)
        ax.set_xlabel('Feature value')
        ax.set_ylabel('')
        ax.spines['left'].set_color('w')
        ax.tick_params(axis='y', length=0)
        fig.tight_layout()

        # Show figure
        fig.show()

