import re


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
