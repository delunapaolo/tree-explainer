# String manipulation
import re
# Numerical
import numpy as np
# Local repository
from .utilities.numerical import divide0


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
    paths = tuple([np.array(path)[::-1] for path in paths])

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
        this_feature_depth = [np.where(np.in1d(path, feature_position))[0]
                              for path in paths]
        # Get unique list of nodes at which the feature can be found
        this_feature_depth = np.unique(np.hstack(this_feature_depth))
        # Get threshold values
        thresholds_this_feature = thresholds[feature_position]
        # Store mean depth, threshold values and number of nodes
        min_feature_depth[i_feature] = this_feature_depth.min()
        no_of_nodes[i_feature] = feature_position.shape[0]
        value_threshold[i_feature] = thresholds_this_feature.mean()

    # Store results
    with lock:
        results['tree_path'][i_tree] = paths
        results['tree_depth'][i_tree] = max([len(i) for i in paths])
        for i_feature, feature in enumerate(features):
            feature_name = 'feature_%i' % (feature + 1)
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


def _compute_feature_contributions_from_tree(estimator, X_test, paths,
                                             compute_conditional_contribution,
                                             results, lock):
    """For a given tree estimator computes target frequency at the tree root
    and feature contributions, such that prediction ≈ target_frequency_at_root +
    feature_contributions.

    REFERENCE: This function is based on _predict_tree in the python package
    treeinterpreter.
    SOURCE: https://github.com/andosa/treeinterpreter/blob/master/treeinterpreter/treeinterpreter.py

    :param estimator: [object] The tree from which to calculate feature
        contributions.
    :param X_test: [numpy array] Data on which to test feature contributions. It
        must have the same number of features of the dataset used to train
        the model.
    :param compute_conditional_contribution: [bool] Whether to also return all
        the conditional contributions along the path.

    :return results: [dict] Contains:
    - predictions: [numpy array] Prediction of each feature to each observation
        and target.
    - target_frequency_at_root: [numpy array] Baseline prediction of each feature
        to each observation and target.
    - contributions: [numpy array] Contribution of each feature to each observation
        and target.
    - conditional_contributions: [dict] (optional if
        `compute_conditional_contribution` == True) A dictionary containing
        the values of contribution of each feature conditioned on previous
        features. Each key contains the list of features along the decision
        path, from the root down to the leaf. Each dictionary value contains
        a numpy array containing the conditional contribution of each
        feature to each observation.
    - conditional_contributions_samples: [dict] (optional if
        `compute_conditional_contribution` == True) A dictionary with the same
        keys of conditional_contributions. Instead of conditional contribution
        values, it contains the index of the observations where the values of
        conditional distributions were calculated. This dictionary could be used
        to select only observations from the true class of interest.
    """

    # Get list of features used at each node
    node_features = estimator.tree_.feature

    # Initialize output variables
    contributions = np.zeros_like(results['contributions']) * np.nan
    # Create the dictionaries containing the actual values of conditional
    # contribution, and the index of the samples that ended up in each array
    conditional_contributions = dict()
    # conditional_contributions_samples = dict()
    # Initialize intermediate variables
    path_to_features = None

    # Retrieve leaves and paths
    leaves_X = estimator.apply(X_test)
    # Map leaves to paths
    leaf_to_path = {path[-1]: path for path in paths}

    # The attribute `value` holds he amount of training samples that end up in
    # the respective node for each class.
    # Reference: https://stackoverflow.com/a/47719621
    n_samples_in_node = estimator.tree_.value.squeeze(axis=1)
    if n_samples_in_node.ndim == 0:
        n_samples_in_node = np.array([n_samples_in_node])

    # Compute target frequency at root of each target
    _, _, implementation, estimator_type = validate_model_type(estimator)
    if estimator_type == 'DecisionTreeClassifier' and implementation == 'sklearn':
        # Convert class count to probabilities
        probabilities_in_node = divide0(n_samples_in_node,
                                        n_samples_in_node.sum(axis=1, keepdims=True),
                                        replace_with=0)
    else:
        probabilities_in_node = n_samples_in_node

    # If the tree did not perform any split, return immediately
    if node_features[node_features >= 0].shape[0] == 0:
        raise Exception
        # if compute_conditional_contribution:
        #     return predictions, target_frequency_at_root, contributions, None, None
        # else:
        #     return predictions, target_frequency_at_root, contributions

    if compute_conditional_contribution:
        # Map each path to the whole conditional set of features up to that node
        path_to_features = {tuple(path): tuple(node_features[path[:-1]]) for
                            path in paths}

    # Compute contributions
    for i_obs, leaf in enumerate(leaves_X):
        path = leaf_to_path[leaf]
        # Compute absolute contribution of each feature at each node (that is, at each split)
        contrib_features = probabilities_in_node[path[1:], :] - probabilities_in_node[path[:-1], :]
        contrib_features_index = node_features[path[:-1]]
        # Store data
        contributions[i_obs, contrib_features_index, :] = np.sum(contrib_features, axis=0)

        # Compute conditional contributions
        if compute_conditional_contribution:
            # Compute incremental contributions down the path, due by conditioning
            # a feature preceding ones in the path
            contrib_features_joint = np.cumsum(contrib_features, axis=0)

            # Store values
            features_at_this_node = path_to_features[tuple(path)]
            if features_at_this_node in conditional_contributions.keys():
                conditional_contributions[features_at_this_node] = \
                    np.dstack((conditional_contributions[features_at_this_node],
                               contrib_features_joint))
                # conditional_contributions_samples[features_at_this_node].append(i_obs)
            else:
                conditional_contributions[features_at_this_node] = contrib_features_joint
                # conditional_contributions_samples[features_at_this_node] = [i_obs]

    # Store results
    with lock:
        # Extract values of contributions, replace NaN with 0 and count number of non-NaN
        n = np.isfinite(contributions).astype(int)
        contributions[np.logical_not(np.isfinite(contributions))] = 0
        results['contributions'] += contributions
        results['contributions_n_evaluations'] += n

        if compute_conditional_contribution:
            # Add conditional contribution values and sample index of each
            # contribution
            feature_sets = list(conditional_contributions.keys())
            for features in feature_sets:
                if features in results['conditional_contributions'].keys():
                    results['conditional_contributions'][features] = \
                        np.dstack((results['conditional_contributions'][features],
                                   conditional_contributions[features]))
                else:
                    results['conditional_contributions'] = conditional_contributions[features]



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
