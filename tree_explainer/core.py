# String manipulation
import re
import warnings
warnings.simplefilter('ignore', FutureWarning)

# Numerical
import numpy as np
import pandas as pd

# Local repository
from tree_explainer.utilities.numerical import divide0


def _compute_tree_paths(estimator, i_tree, results, lock):
    """Get information about a tree model, such as the depth of the tree, the
    depth at which each feature is found, the number of nodes in the tree, and
    the threshold values used for splitting at each node.

    :param estimator: [object] The tree model.
    :param i_tree: [int] The index of the tree currently analyzed.
    :param results: [dict] The output variable, which is shared with the other
        cores.
    :param lock: [object] A threading.Lock() instance.

    :return results: [dict] Contains:
    - feature_depth: [dict] Each key (corresponding to a feature) contains
        the depths in the tree at which it was used for a split. Unused
        features are not listed.
    - tree_depth: [int] The depth of this tree, which equals the length of
        the longest path from root to leaves.
    - no_of_nodes: [dict] Each key (corresponding to a feature) contains
        the number of nodes in which the feature appears for splitting.
        Unused features are not listed.
    - value_threshold: [dict] Each key (corresponding to a feature) contains
        the threshold for splitting on a feature. Unused features are not
        listed.
    """

    paths = get_tree_paths(estimator.tree_, node_id=0, depth=0)
    # Reverse direction of path, and convert to tuple
    paths = tuple([np.array(branch)[::-1] for branch in paths])

    # Get list of the feature used at each node
    node_features = estimator.tree_.feature
    # Get list of thresholds used for each feature
    thresholds = estimator.tree_.threshold

    # Get the depth at which each feature is encountered, and the threshold
    # value used for the split
    features = np.unique(node_features[node_features >= 0])
    n_features = features.shape[0]
    min_feature_depth = np.zeros((n_features, ), dtype=np.int8)
    no_of_nodes = np.zeros((n_features, ), dtype=np.int8)
    value_threshold = np.zeros((n_features, ), dtype=np.float32)
    for i_feature, feature in enumerate(features):
        feature_position = np.where(node_features == feature)[0]
        # Look for the position of this feature in each path
        this_feature_depth = [np.where(np.in1d(branch, feature_position))[0]
                              for branch in paths]
        # Get unique list of nodes at which the feature can be found
        this_feature_depth = np.unique(np.hstack(this_feature_depth))
        # Get threshold values
        thresholds_this_feature = thresholds[feature_position]
        # Store minimal depth, threshold values and number of nodes
        min_feature_depth[i_feature] = this_feature_depth.min()
        no_of_nodes[i_feature] = feature_position.shape[0]
        value_threshold[i_feature] = thresholds_this_feature.mean()

    # Get list of conditional set of features up to each node
    nodes_to_features = tuple([node_features[branch[:-1]] for branch in paths])
    # Get pair of parent-child feature at each node in each branch
    features_split = list()
    for i_path, features_this_path in enumerate(nodes_to_features):
        features_split.extend(np.vstack(([-1] + [features_this_path[i] for i in range(len(features_this_path) - 1)],
                                         [features_this_path[i] for i in range(len(features_this_path))],
                                         paths[i_path][:-1])).T)
    # Convert to a pandas DataFrame, with unique rows
    features_split = pd.DataFrame(np.unique(np.vstack(features_split).astype(int), axis=0), columns=['parent', 'child', 'node_idx'])

    # Store results
    with lock:
        results['tree_path'][i_tree] = paths
        results['tree_feature_path'][i_tree] = nodes_to_features
        results['tree_depth'][i_tree] = max([len(i) for i in paths])
        results['features_split'][i_tree] = features_split
        for i_feature, feature in enumerate(features):
            feature_name = results['feature_names'][feature]
            results['feature_depth'][feature_name][i_tree] = min_feature_depth[i_feature]
            results['value_threshold'][feature_name][i_tree] = value_threshold[i_feature]
            results['no_of_nodes'][feature_name][i_tree] = no_of_nodes[i_feature]


def get_tree_paths(tree, node_id, depth=0):
    """Recursively navigate a tree model to gather the node_ids of each decision
    path.

    REFERENCE: This function is based on _get_tree_paths in the python package
    treeinterpreter.
    SOURCE: https://github.com/andosa/treeinterpreter/blob/master/treeinterpreter/treeinterpreter.py
    """

    # The following variables mark leaves and undefined nodes. They can be
    # obtained with: from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
    TREE_LEAF = -1
    # For the records, TREE_UNDEFINED = -2

    if node_id == TREE_LEAF:
        raise ValueError("Invalid node_id %s" % node_id)

    # Get the id of the left branch
    left_branch = tree.children_left[node_id]
    if left_branch != TREE_LEAF:  # we haven't reached a leaf yet
        # Follow path a level deeper on the left and the right branches
        left_paths = get_tree_paths(tree, left_branch, depth=depth + 1)
        right_branch = tree.children_right[node_id]
        right_paths = get_tree_paths(tree, right_branch, depth=depth + 1)
        # Append the if of the current node to all paths leading here
        for path in left_paths:
            path.append(node_id)
        for path in right_paths:
            path.append(node_id)
        # Append the two lists
        paths = left_paths + right_paths

    else:  # This path has led to a leaf
        paths = [[node_id]]

    return paths


def _compute_feature_contributions_from_tree(estimator, i_tree, X, y,
                                             paths, compute_conditional_contributions,
                                             results, lock):
    """For a given tree estimator computes target frequency at the tree root
    and feature contributions, such that prediction ≈ target_frequency_at_root +
    feature_contributions.

    REFERENCE: This function is based on _predict_tree in the python package
    treeinterpreter.
    SOURCE: https://github.com/andosa/treeinterpreter/blob/master/treeinterpreter/treeinterpreter.py

    :param estimator: [object] The tree from which to calculate feature
        contributions.
    :param i_tree: [int] The index of the tree in the ensemble model.
    :param X: [numpy array] Data on which to test feature contributions. It
        must have the same number of features of the dataset used to train
        the model.
    :param compute_conditional_contributions: [bool] Whether to also return all
        the conditional contributions along the path.

    :return results: [dict] Contains:
    - predictions: [numpy array] Prediction of each feature to each observation
        and target.
    - target_frequency_at_root: [numpy array] Baseline prediction of each feature
        to each observation and target.
    - contributions: [numpy array] Contribution of each feature to each observation
        and target.
    - conditional_contributions: [dict] (optional if
        `compute_conditional_contributions` == True) A dictionary containing
        the values of contribution of each feature conditioned on previous
        features. The key is the index of the decision path. Each value contains
        a numpy array of the conditional contribution of each feature to each
        observation in that node.
    - conditional_contributions_samples: [dict] (optional if
        `compute_conditional_contributions` == True) A dictionary with the same
        keys of conditional_contributions. It contains the index of the
        observations where the values of conditional contributions are calculated.
        This information can be used to select subsets of observations in later
        analyses.
    """

    # Get list of features used at each node
    node_features = estimator.tree_.feature
    n_nodes = node_features.shape[0]
    # If the tree did not perform any split, do not compute conditional contributions
    n_splits = node_features[node_features >= 0].shape[0]
    if n_splits < 1:
        compute_conditional_contributions = False

    # Initialize output variables
    contributions = np.zeros_like(results['contributions'])
    n_values = np.zeros((contributions.shape[1], ), dtype=int)

    # Initialize intermediate variables
    conditional_contributions = None
    conditional_contributions_lookup = None
    conditional_contributions_sample = None
    features_split = None

    # Retrieve leaves and paths
    leaves_X = estimator.tree_.apply(X)
    # Map leaves to paths
    leaf_to_path = {path[-1]: path for path in paths}

    # The attribute `value` holds he amount of training samples that end up in
    # the respective node for each class (reference:
    # https://stackoverflow.com/a/47719621). Instead of the following code,
    # which works for training data only:
    # n_samples_in_node = estimator.tree_.value.squeeze(axis=1)
    # if n_samples_in_node.ndim == 0:
    #     n_samples_in_node = np.array([n_samples_in_node])
    # we'll use the `predict` method, which applies to dataset `X`. However, we
    # need to split `X` by `y` to obtain the number of samples in each class that
    # passed through each node.
    targets = np.unique(y)
    n_targets = len(targets)
    n_samples_in_node = np.zeros((n_nodes, n_targets), dtype=float)
    for i_col, target in enumerate(targets):
        this_X = X[y == target, :]
        decision_path = estimator.tree_.decision_path(this_X)
        n_samples_in_node[:, i_col] = np.sum(decision_path, axis=0)

    # Compute target frequency at root of each target
    _, _, implementation, estimator_type = validate_model_type(estimator)
    if estimator_type == 'DecisionTreeClassifier' and implementation == 'sklearn':
        # Convert class count to probabilities
        probabilities_in_node = divide0(n_samples_in_node,
                                        n_samples_in_node.sum(axis=1, keepdims=True),
                                        replace_with=0)
    else:
        probabilities_in_node = n_samples_in_node.copy()

    # Get the probability of each target at the root
    target_probability_at_root = probabilities_in_node[paths[0][0]]

    if compute_conditional_contributions:
        # Make arrays for conditional contributions
        features_split = results['features_split'][i_tree].copy()
        no_of_nodes = features_split.shape[0]
        conditional_contributions = np.zeros((no_of_nodes, n_targets),
                                             dtype=X.dtype) * np.nan
        conditional_contributions_lookup = np.zeros((no_of_nodes, ), dtype=bool)
        conditional_contributions_sample = dict()

        # Convert DataFrame index for faster searching
        features_split['row'] = features_split.index
        features_split.set_index(features_split['node_idx'], inplace=True)

    # Compute contributions
    for i_obs, leaf in enumerate(leaves_X):
        # Get path where this leaf is located
        path = leaf_to_path[leaf].copy()
        # Get final prediction
        prediction = probabilities_in_node[leaf]
        # Get values of probabilities at each node
        contrib_features = probabilities_in_node[path, :].copy()

        # Look for node where we get the closest to the prediction
        stop_criterion = np.where(contrib_features[:, prediction.argmax()].astype(int) == 1)[0]
        if stop_criterion.shape[0] > 0:
            # Take earliest node where certainty was reached
            stop_criterion = stop_criterion[0] + 1
            # Prune current branch
            contrib_features = contrib_features[:stop_criterion, :]
            path = path[:stop_criterion]

        # Get the indices of the nodes where this path is located
        nodes_idx = path[:-1]

        # Compute change in probability of each target at each node split along
        # the current path. The sum of these values along the path and
        # `target_probability_at_root` should approximate well the values of
        # `prediction`.
        delta_contributions = contrib_features[1:, :] - contrib_features[:-1, :]
        # Get indices of features participating at each node in this path
        features_index = node_features[path[:-1]]
        # Store data. np.add.at() is equivalent to a[indices] += b, except that
        # results are accumulated for elements that are indexed more than once
        np.add.at(contributions[i_obs, :, :], features_index, delta_contributions)
        np.add.at(n_values, features_index, 1)

        # Compute conditional contributions
        if compute_conditional_contributions:
            # Store index of nodes for this observation
            conditional_contributions_sample[i_obs] = features_split.loc[nodes_idx, 'row'].values

            # Continue only if there are nodes that we haven't stored
            analyzed_nodes = conditional_contributions_lookup[features_split.loc[nodes_idx, 'row']]
            if not analyzed_nodes.all():
                # Get rows of nodes to analyze
                nodes_to_analyze_idx = np.where(np.logical_not(analyzed_nodes))[0]
                rows = features_split.loc[nodes_idx, 'row'].values[nodes_to_analyze_idx]

                # Transform values into how much (in percentage) each split
                # contributes to final prediction
                distance_from_baseline_to_prediction = prediction - target_probability_at_root
                conditional_contributions[rows, :] = divide0(delta_contributions[nodes_to_analyze_idx, :],
                                                             distance_from_baseline_to_prediction * 100,
                                                             replace_with=0)
                # Toggle that we have analyzed these nodes
                conditional_contributions_lookup[rows] = True

    # Store results
    with lock:
        # Extract values of contributions, replace NaN with 0 and count number
        # of non-NaNs
        n = np.isfinite(contributions).astype(int)
        contributions[np.logical_not(np.isfinite(contributions))] = 0
        results['contributions'] += contributions
        results['contributions_n_evaluations'] += n

        if compute_conditional_contributions:
            # Store conditional contributions
            results['conditional_contributions'][i_tree] = conditional_contributions
            results['conditional_contributions_sample'][i_tree] = conditional_contributions_sample


def _compute_two_way_conditional_contributions(samples, conditional_contributions,
                                               conditional_contributions_sample,
                                               features_split,
                                               results, lock):

    # Initialize heatmap-tables
    dims = results['contribution_values'].shape
    n_features = dims[0]
    contribution_values = np.zeros(dims)
    contribution_values_n = np.zeros_like(results['contribution_values_n'])

    for this_sample in samples:
        # Get the contribution values for this observation
        rows = conditional_contributions_sample[this_sample]
        this_contribution_values = conditional_contributions[rows, :]

        # Find which features generated these values. These indices
        # are coordinated in the heatmap, that is, children on rows,
        # parents on columns
        coords = features_split.loc[rows, ['child', 'parent']].values.T.tolist()
        # Store values. np.add.at() is equivalent to a[indices] += b,
        # except that results are accumulated for elements that are
        # indexed more than once
        np.add.at(contribution_values, coords, this_contribution_values)
        np.add.at(contribution_values_n, coords, 1)

    # Store results
    with lock:
        results['contribution_values'] += contribution_values
        results['contribution_values_n'] += contribution_values_n


def validate_model_type(model):
    """Get information on this model. If not recognized this function returns a
    NotImplementedError error.

    :param model: The input model.
    :return model_class: [str] The model class (e.g., RandomForestClassifier).
    :return model_type: [str] The type of model (i.e., 'classifier' or
        'regressor').
    :return implementation: [str] The package implementing the model (e.g.,
        sklearn).
    :return estimator_type: [str] The class of the estimator (e.g.,
        DecisionTreeClassifier).
    """

    # List of known model types
    KNOWN_MODEL_TYPES = [
        # RandomForests
        'sklearn.ensemble.forest.RandomForestClassifier',
        'sklearn.ensemble.forest.RandomForestRegressor',
        # Ensemble of extremely randomized trees
        'sklearn.ensemble.forest.ExtraTreesClassifier',
        'sklearn.ensemble.forest.ExtraTreesRegressor',

        # Decision trees
        'sklearn.tree.tree.DecisionTreeClassifier',
        'sklearn.tree.tree.DecisionTreeRegressor',
        # Extremely randomized trees
        'sklearn.tree.tree.ExtraTreeClassifier',
        'sklearn.tree.tree.ExtraTreeRegressor'
        ]

    # The class of the current model
    model_class = str(type(model))
    # Check class against known types
    result = re.search('\'(.*)\'', model_class)
    model_class_str = result.group(1)
    if model_class_str in KNOWN_MODEL_TYPES:
        # Infer class and package of origin of the model
        model_class_parts = model_class_str.split('.')
        model_class = model_class_parts[-1]
        implementation = model_class_parts[0]

        # Get type of estimator
        if model_class == 'RandomForestClassifier':
            estimator_type = 'DecisionTreeClassifier'
        elif model_class == 'RandomForestRegressor':
            estimator_type = 'DecisionTreeRegressor'
        elif model_class == 'ExtraTreesClassifier':
            estimator_type = 'ExtraTreeClassifier'
        elif model_class == 'ExtraTreesRegressor':
            estimator_type = 'ExtraTreeRegressor'
        else:
            estimator_type = model_class

        # Get type of model
        if 'classifier' in estimator_type.lower():
            model_type = 'classifier'
        elif 'regressor' in estimator_type.lower():
            model_type = 'regressor'
        else:
            raise NotImplementedError('Not clear whether \'%s\' is a classifier or a regressor')

        return model_class, model_type, implementation, estimator_type

    else:
        raise NotImplementedError('Class \'%s\' is not supported by TreeExplainer' % model_class_str)


def validate_model_is_trained(model, model_type):
    """Check whether the model has been already trained.

    :param model: The input model.
    :param model_type: [str] The model type (e.g., RandomForestClassifier).
    :return Either nothing or an AttributeError.
    """
    if model_type in ['RandomForestClassifier', 'RandomForestRegressor',
                      'ExtraTreesClassifier', 'ExtraTreesRegressor']:
        n_estimators_attr = 'n_estimators'
        estimators_attr = 'estimators_'

    else:
        raise NotImplementedError('Don\'t know what to do with \'%s\'' % model_type)

    if hasattr(model, n_estimators_attr) and not hasattr(model, estimators_attr):
        raise AttributeError('The model has not been trained yet, and thus cannot be explained.')

    else:
        return n_estimators_attr, estimators_attr
