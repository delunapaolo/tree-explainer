# String manipulation
import warnings
warnings.simplefilter('ignore', FutureWarning)

# Numerical
import numpy as np
import pandas as pd

# Local repository
from tree_explainer.utilities.model_validation import validate_model_type
from tree_explainer.utilities.numerical import divide0

# The following variables mark leaves and undefined nodes. They can be
# obtained with: from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
TREE_LEAF = -1
TREE_UNDEFINED = -2


def analyze_tree_structure(estimator, feature_names, store_tree_path, i_tree, results, lock):
    """For a given tree estimator computes target frequency at the tree root
    and feature contributions, such that prediction ≈ target_frequency_at_root +
    feature_contributions.

    REFERENCE: This function is based on _predict_tree in the python package
    treeinterpreter.
    SOURCE: https://github.com/andosa/treeinterpreter/blob/master/treeinterpreter/treeinterpreter.py

    :param estimator: [object] The tree from which to calculate feature
        contributions.
    :param feature_names: [list] Names of features.
    :param store_tree_path: [bool]
    :param i_tree: [int] The index of the tree in the ensemble model.
    :param results:
    :param lock:

    :return results: [dict] Contains:
    - feature_depth: [dict] Each key (corresponding to a feature) contains
        the depths in the tree at which it was used for a split. Unused
        features are not listed.
    - tree_depth: [int] The depth of this tree, which equals the length of
        the longest path from root to leaves.
    - no_of_nodes: [dict] Each key (corresponding to a feature) contains
        the number of nodes in which the feature appears for splitting.
        Unused features are not listed.
    - threshold_value: [dict] Each key (corresponding to a feature) contains
        the threshold for splitting on a feature. Unused features are not
        listed.
    """

    # Tree structure
    paths, paths_sign = _get_tree_paths(estimator.tree_, node_id=0, depth=0)

    # Reverse direction of path, and convert to tuple
    paths = tuple([np.array(branch)[::-1] for branch in paths])

    # Get list of the feature used at each node
    node_features = estimator.tree_.feature
    # Get list of thresholds used for each feature
    thresholds = estimator.tree_.threshold
    # Get list of left and right children
    left_children = estimator.tree_.children_left
    right_children = estimator.tree_.children_right

    # Get the depth at which each feature is encountered, and the threshold
    # value used for the split
    features = np.unique(node_features[node_features >= 0])
    n_features = features.shape[0]
    min_feature_depth = np.zeros((n_features,), dtype=np.int8)
    no_of_nodes_feature = np.zeros((n_features,), dtype=np.int8)
    for i_feature, feature in enumerate(features):
        feature_position = np.where(node_features == feature)[0]
        # Look for the position of this feature in each path
        this_feature_depth = np.hstack([np.where(np.in1d(branch, feature_position))[0] for branch in paths])
        # Store minimal depth, threshold values and number of nodes
        min_feature_depth[i_feature] = this_feature_depth.min()
        no_of_nodes_feature[i_feature] = feature_position.shape[0]

    # Get list of conditional set of features up to each node
    nodes_to_features = tuple([node_features[branch[:-1]] for branch in paths])
    # Get pair of parent-child feature at each node in each branch
    features_split = list()
    for i_path, features_this_path in enumerate(nodes_to_features):
        features_split.extend(np.vstack(([-1] + [features_this_path[i] for i in range(len(features_this_path) - 1)],
                                         [features_this_path[i] for i in range(len(features_this_path))],
                                         paths[i_path][:-1])).T)
    # Convert to pandas DataFrame, with unique rows
    features_split = pd.DataFrame(np.unique(np.vstack(features_split).astype(int), axis=0),
                                  columns=['parent', 'child', 'node_idx'])
    # Convert index for faster searching
    features_split['row'] = features_split.index
    features_split.set_index(features_split['node_idx'], inplace=True)
    features_split.drop(columns=['node_idx'], inplace=True)

    # Make a DataFrame of decisions made at each node, based on feature values
    threshold_value = pd.DataFrame(np.vstack((node_features, thresholds, left_children, right_children)).T,
                                   columns=['feature_for_split', 'value', 'node_if_less_than', 'node_if_more_than'])
    # Remove nodes corresponding to leaves
    threshold_value.drop(index=np.where(thresholds == TREE_UNDEFINED)[0], inplace=True)
    # Cast to correct data type
    threshold_value['feature_for_split'] = threshold_value['feature_for_split'].map(int)
    threshold_value['node_if_less_than'] = threshold_value['node_if_less_than'].map(int)
    threshold_value['node_if_more_than'] = threshold_value['node_if_more_than'].map(int)
    # Current index of DataFrame is the node id
    threshold_value['node_id'] = threshold_value.index
    threshold_value.reset_index(drop=True, inplace=True)

    # Store results
    with lock:
        if store_tree_path:
            results['tree_path'][i_tree] = paths
        results['features_split'][i_tree] = features_split
        results['tree_feature_path'][i_tree] = nodes_to_features
        results['tree_depth'][i_tree] = max([len(i) for i in paths])
        results['threshold_value'][i_tree] = threshold_value

        for i_feature, feature in enumerate(features):
            results['feature_depth'][feature_names[feature]][i_tree] = min_feature_depth[i_feature]
            results['no_of_nodes'][feature_names[feature]][i_tree] = no_of_nodes_feature[i_feature]


def compute_feature_contributions_from_tree(estimator, data, contributions_shape,
                                            features_split,
                                            joint_contributions,
                                            ignore_non_informative_nodes):
    """For a given tree estimator computes target frequency at the tree root
    and feature contributions, such that prediction ≈ target_frequency_at_root +
    feature_contributions.

    REFERENCE: This function is based on _predict_tree in the python package
    treeinterpreter.
    SOURCE: https://github.com/andosa/treeinterpreter/blob/master/treeinterpreter/treeinterpreter.py

    :param estimator: [object] The tree from which to calculate feature
        contributions.
    :param data: [numpy array] Data on which to test feature contributions. It
        must have the same number of features of the dataset used to train
        the model.
    :param contributions_shape: [tuple]
    :param features_split:
    :param joint_contributions: [bool] Whether to also return all
        the conditional contributions along the path.
    :param ignore_non_informative_nodes:

    :return results: [dict] Contains:
    - contributions: [numpy array] Contribution of each feature to each observation
        and target.
    - conditional_contributions: [dict] (optional if `joint_contributions` ==
        True) A dictionary containing the values of contribution of each feature
        conditioned on previous features. The key is the index of the decision
        path. Each value contains a numpy array of the conditional contribution
        of each feature to each observation in that node.
    - conditional_contributions_samples: [dict] (optional if `joint_contributions`
        == True) A dictionary with the same keys of conditional_contributions.
        It contains the index of the observations where the values of conditional
        contributions are calculated. This information can be used to select
        subsets of observations in later analyses.
    """

    # Tree structure
    paths, paths_sign = _get_tree_paths(estimator.tree_, node_id=0, depth=0)

    # Reverse direction of path, and convert to tuple
    paths = tuple([np.array(branch)[::-1] for branch in paths])

    if joint_contributions:
        if features_split is None:
            # Get list of the feature used at each node
            node_features = estimator.tree_.feature

            # Get list of conditional set of features up to each node
            nodes_to_features = tuple([node_features[branch[:-1]] for branch in paths])
            # Get pair of parent-child feature at each node in each branch
            features_split = list()
            for i_path, features_this_path in enumerate(nodes_to_features):
                features_split.extend(np.vstack(([-1] + [features_this_path[i] for i in range(len(features_this_path) - 1)],
                                                [features_this_path[i] for i in range(len(features_this_path))],
                                                 paths[i_path][:-1])).T)
            # Convert to pandas DataFrame, with unique rows
            features_split = pd.DataFrame(np.unique(np.vstack(features_split).astype(int), axis=0),
                                         columns=['parent', 'child', 'node_idx'])
            # Convert index for faster searching
            features_split['row'] = features_split.index
            features_split.set_index(features_split['node_idx'], inplace=True)
            features_split.drop(columns=['node_idx'], inplace=True)

    # Get indices of leaves where observations would end
    leaves_X = estimator.tree_.apply(data)
    # Map leaves to paths
    leaf_to_path = {path[-1]: path for path in paths}

    # The attribute `value` holds the amount of training samples that end up in
    # the respective node for each class (reference:
    # https://stackoverflow.com/a/47719621)
    n_samples_in_node = estimator.tree_.value.squeeze(axis=1)
    if n_samples_in_node.ndim == 0:
        n_samples_in_node = np.array([n_samples_in_node])

    # Convert class count to probabilities
    _, _, implementation, estimator_type = validate_model_type(estimator)
    if estimator_type == 'DecisionTreeClassifier' and implementation == 'sklearn':
        probabilities_in_node = divide0(n_samples_in_node,
                                        n_samples_in_node.sum(axis=1, keepdims=True),
                                        replace_with=0)
    else:
        probabilities_in_node = n_samples_in_node.copy()

    # Get the probability of each target at the root
    target_probability_at_root = probabilities_in_node[paths[0][0]]

    # Get list of features used at each node
    node_features = estimator.tree_.feature
    # n_nodes = node_features.shape[0]
    # If the tree did not perform any split, do not compute conditional contributions
    n_splits = node_features[node_features >= 0].shape[0]
    if n_splits < 1:
        joint_contributions = False

    # Initialize output variables
    contributions = np.zeros(contributions_shape, dtype=np.float64)
    n_values = np.zeros((contributions.shape[1], ), dtype=int)
    conditional_contributions = None
    conditional_contributions_lookup = None
    conditional_contributions_sample = None

    if joint_contributions:
        # Make arrays for conditional contributions
        no_of_nodes = features_split.shape[0]
        n_targets = target_probability_at_root.shape[0]
        conditional_contributions = np.zeros((no_of_nodes, n_targets)) * np.nan
        conditional_contributions_lookup = np.zeros((no_of_nodes, ), dtype=bool)
        conditional_contributions_sample = dict()

    # Compute contributions
    for i_obs, leaf in enumerate(leaves_X):
        # Get path where this leaf is located
        path = leaf_to_path[leaf].copy()
        # Get final prediction
        prediction = probabilities_in_node[leaf]
        # Get values of probabilities at each node
        contrib_features = probabilities_in_node[path, :]

        # Compute change in probability of each target at each node split along
        # the current path. The sum of these values along the path and
        # `target_probability_at_root` should approximate well the values of
        # `prediction`.
        delta_contributions = contrib_features[1:, :] - contrib_features[:-1, :]

        if ignore_non_informative_nodes:
            # It can happen that node splits do not contribute to predict the values
            # of this observation. In that case, contribution values for all targets
            # are 0s
            uninformative_nodes = np.all(delta_contributions == 0, axis=1)
            if np.any(uninformative_nodes):
                delta_contributions = np.delete(delta_contributions, np.where(uninformative_nodes)[0], axis=0)

                # Shift all indices by 1 forward because `uninformative_nodes` refers
                # to a difference
                nodes_to_keep = np.where(np.logical_not(np.hstack(([False], uninformative_nodes))))[0]
                # Replace last informative node with the actual leaf
                nodes_to_keep[-1] = path.shape[0] - 1
                # Prune current branch, removing uninformative nodes
                path = path[nodes_to_keep]

        # Get indices of features participating at each node in this path
        features_index = node_features[path[:-1]]

        # Store data. np.add.at() is equivalent to a[indices] += b, except
        # that results are accumulated for elements that are indexed more
        # than once
        np.add.at(contributions[i_obs, :, :], features_index, delta_contributions)
        np.add.at(n_values, features_index, 1)

        if joint_contributions:
            # Get the indices of the nodes where this path is located
            nodes_idx = path[:-1]
            # Store index of nodes for this observation
            nodes_rows = features_split.loc[nodes_idx, 'row'].values
            conditional_contributions_sample[i_obs] = nodes_rows

            # Continue only if there are nodes that we haven't stored
            analyzed_nodes = conditional_contributions_lookup[nodes_rows]
            if not analyzed_nodes.all():
                # Get rows of nodes to analyze
                nodes_to_analyze_idx = np.where(np.logical_not(analyzed_nodes))[0]
                rows = nodes_rows[nodes_to_analyze_idx]

                # Transform values into how much (in percentage) each split
                # contributes to final prediction
                distance_from_baseline_to_prediction = prediction - target_probability_at_root
                conditional_contributions[rows, :] = divide0(delta_contributions[nodes_to_analyze_idx, :],
                                                             distance_from_baseline_to_prediction,
                                                             replace_with=0) * 100
                # Toggle that we have analyzed these nodes
                conditional_contributions_lookup[rows] = True

    # Store results
    results = dict()
    results['tree_path'] = paths
    results['data_leaves'] = leaves_X
    results['target_probability_at_root'] = target_probability_at_root

    # Extract values of contributions, replace NaN with 0 and count number
    # of non-NaNs
    n = np.isfinite(contributions).astype(int)
    contributions[np.logical_not(np.isfinite(contributions))] = 0
    results['contributions'] = contributions
    results['contributions_n_evaluations'] = n

    if joint_contributions:
        # Store conditional contributions
        results['conditional_contributions'] = conditional_contributions
        results['conditional_contributions_sample'] = conditional_contributions_sample
        results['features_split'] = features_split

    return results


def _get_tree_paths(tree, node_id, depth=0):
    """Recursively navigate a tree model to gather the node_ids of each decision
    path.

    REFERENCE: This function is based on _get_tree_paths in the python package
    treeinterpreter.
    SOURCE: https://github.com/andosa/treeinterpreter/blob/master/treeinterpreter/treeinterpreter.py
    """

    if node_id == TREE_LEAF:
        raise ValueError('Invalid node_id %s' % node_id)

    # Get the id of the left branch
    left_branch = tree.children_left[node_id]
    if left_branch != TREE_LEAF:  # we haven't reached a leaf yet
        # Follow path a level deeper on the left and the right branches
        left_paths, left_paths_sign = _get_tree_paths(tree, left_branch, depth=depth + 1)
        right_branch = tree.children_right[node_id]
        right_paths, right_paths_sign = _get_tree_paths(tree, right_branch, depth=depth + 1)
        # Append the if of the current node to all paths leading here
        for path in left_paths:
            path.append(node_id)
        for path in right_paths:
            path.append(node_id)
        # Mark left and right children in path
        for path in left_paths_sign:
            path.append(0)
        for path in right_paths_sign:
            path.append(1)
        # Append the two lists
        paths = left_paths + right_paths
        paths_sign = left_paths_sign + right_paths_sign

    else:  # This path has led to a leaf
        paths = [[node_id]]
        paths_sign = [[-1]]

    return paths, paths_sign


def compute_two_way_conditional_contributions(samples, conditional_contributions,
                                              conditional_contributions_sample,
                                              features_split,
                                              results, lock):

    # Initialize heatmap-tables
    dims = results['contribution_values'].shape
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


def compute_explanation_of_prediction(leaf, paths, threshold_value, features_split,
                                      conditional_contributions, prediction,
                                      feature_values, solve_duplicate_splits):
    """

    :param leaf:
    :param paths:
    :param threshold_value:
    :param features_split:
    :param conditional_contributions:
    :param prediction:
    :param feature_values:
    :param solve_duplicate_splits:
    :param results:
    :param lock:

    :return:
    """

    # Get names of features
    feature_names = list(feature_values.keys())
    # Initialize output variables
    samples_for_decision_table = dict({i: list() for i in feature_names})
    contributions_for_decision_table = dict({i: list() for i in feature_names})

    # Get the tree path which this observation followed
    path_idx = np.where([np.any(np.in1d(path, leaf)) for path in paths])[0][0]
    path = paths[path_idx]

    # Get the sign of the operations performed at the nodes in which the
    # observation was evaluated
    rows = np.hstack([np.where(threshold_value['node_id'] == i)[0] for i in path[:-1]])
    decision_nodes = threshold_value.loc[rows, ['node_if_less_than', 'node_if_more_than']].values
    signs = np.hstack([np.where(i == j)[0] for i, j in zip(decision_nodes, path[1:])])

    # Gather feature values that satisfy that condition
    values = threshold_value.loc[rows].copy()
    values['sign'] = signs
    values.reset_index(drop=True, inplace=True)

    # Add values of contribution
    values['nodes_followed'] = [i[0] if i[2] == 0 else i[1] for i in values[
        ['node_if_less_than', 'node_if_more_than', 'sign']].values]
    nodes_followed = np.hstack((features_split.loc[0, 'row'], values['nodes_followed'][:-1].values))
    cond_rows = features_split.loc[nodes_followed, 'row']
    values['contribution'] = conditional_contributions[cond_rows, int(prediction)]
    values.dropna(axis=0, how='any', inplace=True)
    values.reset_index(drop=True, inplace=True)

    # If a feature was evaluated more than once, apply a rule to select
    # which piece of information to keep
    if solve_duplicate_splits == 'none':
        pass
    elif solve_duplicate_splits == 'closest':
        pass
    elif solve_duplicate_splits == 'mean':
        pass

    # Make a tuple of the two comparison functions that have been used at each
    # node split, that is, "less than or equal to" and "greater than". Below,
    # the indices 0 and 1 will be assigned to these operations, which can be
    # used to retrieve the correct function
    comparisons = tuple((np.less_equal, np.greater))

    # Store values of each node
    for i_node in range(values.shape[0]):
        row_data = values.loc[i_node]
        # Get the appropriate function for comparison
        this_comparison = comparisons[int(row_data['sign'])]
        # Get the boolean indices where the comparison is True
        feature_idx = int(row_data['feature_for_split'])
        feature_name = feature_names[feature_idx]
        feature_values_idx = this_comparison(feature_values[feature_name], row_data['value'])

        # Store values
        samples_for_decision_table[feature_name].extend(feature_values[feature_name][feature_values_idx])
        contributions_for_decision_table[feature_name].append(row_data['contribution'])

    # Store results
    results = dict(samples_for_decision_table=samples_for_decision_table,
                   contributions_for_decision_table=contributions_for_decision_table)

    return results
