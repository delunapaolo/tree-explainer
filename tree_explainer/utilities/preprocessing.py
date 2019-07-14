import numpy as np
import pandas as pd


class DataProcessor(object):
    def __init__(self):
        """

        """

        self.info = dict()
        self.data = None
        self.targets = None


    def prepare(self, data=None, targets=None, data_column_names=None,
                target_name=None, one_hot_encoding_threshold=3):
        """

        :param data:
        :param targets:
        :param data_column_names:
        :param target_name:
        :param one_hot_encoding_threshold:
        :return:
        """
        # Check that there is something to do
        if data is None and targets is None:
            raise ValueError('Provide data and/or targets')

        if data is not None:
            self._process_data(data, data_column_names, one_hot_encoding_threshold)

        if targets is not None:
            self._process_targets(targets, target_name)

        return self


    def _process_data(self, data, data_column_names, one_hot_encoding_threshold):
        # Initialize output variable
        info = dict()

        # Get feature names
        if isinstance(data, (pd.DataFrame, pd.Series)):
            if isinstance(data, pd.Series):
                raise NotImplementedError
            # Infer feature names
            data_column_names = list(data.columns)
            # Make local copy
            data = data.copy()
            data.reset_index(drop=True, inplace=True)
        else:
            if data_column_names is None:
                data_column_names = ['variable_%i' % (i + 1) for i in range(data.shape[1])]
            # Convert to a DataFrame
            data = pd.DataFrame(data, columns=data_column_names)
        # Store feature names
        info['original_feature_names'] = data_column_names

        # Retrieve information on categorical features
        features_data_types = dict()
        for feature in data_column_names:
            # Get description of data type
            dtype = data.dtypes[feature]
            if dtype.name == 'category':  # == pd.Categorical
                categories = np.array(data[feature].cat.categories)
                n_levels = categories.shape[0]
                if dtype.ordered:
                    data_type = 'ordinal'
                    encoding = 'ordinal'
                else:
                    data_type = 'nominal'
                    if n_levels <= one_hot_encoding_threshold:
                        encoding = 'onehot'
                    else:
                        encoding = 'ordinal'

            elif dtype.name == 'object':  # Most likely a string
                categories = np.unique(data[feature])
                n_levels = categories.shape[0]
                data_type = 'nominal'
                if n_levels <= one_hot_encoding_threshold:
                    encoding = 'onehot'
                else:
                    encoding = 'ordinal'

            else:
                data_type = 'numerical'
                categories = None
                n_levels = None
                encoding = None

            # Store information
            features_data_types[feature] = dict(data_type=data_type,
                                                categories=categories,
                                                n_levels=n_levels,
                                                encoding=encoding)

        # Mark nominal features to be one-hot encoded if have less than
        # `one_hot_encoding_threshold` categories
        variables_to_onehot_encode = [i for i in features_data_types.keys()
                                      if features_data_types[i]['encoding'] == 'onehot']
        # The rest of the variables will be encoded with numbers
        variables_to_numerically_encode = [i for i in features_data_types.keys()
                                           if features_data_types[i]['encoding'] == 'ordinal']

        # Perform the encoding
        if len(variables_to_onehot_encode) > 0:
            list_of_features = list(data.columns)
            # Call pd.get_dummies() on individual columns
            for feature in variables_to_onehot_encode:
                # Check whether this feature contains NaN. If so, we'll add a
                # new column for that
                dummy_na = data[feature].isna().any()
                # Create new features
                engineered_feature = pd.get_dummies(data[feature],
                                                    prefix=feature, prefix_sep='_',
                                                    dummy_na=dummy_na, drop_first=True)
                # Swap information on features
                old_feature_info = features_data_types.pop(feature)
                for new_feature in list(engineered_feature.columns):
                    new_feature_info = old_feature_info.copy()
                    new_feature_info['parent_feature'] = feature
                    features_data_types[new_feature] = new_feature_info
                # Append to data
                data.drop(columns=feature, inplace=True)
                data = pd.concat((data, engineered_feature), axis=1, sort=False, copy=False)
                # Replace feature in list
                idx = list_of_features.index(feature)
                list_of_features.remove(feature)
                list_of_features.insert(idx, list(engineered_feature.columns))

            # Re-sort columns in data
            data = data[np.hstack(list_of_features)]

        if len(variables_to_numerically_encode) > 0:
            for feature in variables_to_numerically_encode:
                categories = features_data_types[feature]['categories']
                categories_lookup = dict({i: j for i, j in zip(categories, np.arange(categories.shape[0]))})
                data[feature] = data[feature].replace(categories_lookup)

        # Store information
        info['features_data_types'] = features_data_types
        info['feature_names'] = list(data.columns)
        info['n_features'] = len(info['feature_names'])

        # Store all information and converted X
        self.data = data.values.astype(np.float32)
        self.info.update(info)


    def _process_targets(self, targets, target_name):
        # Initialize output variable
        info = dict()

        # Get names of targets
        n_target_levels = np.unique(targets).shape[0]
        info['n_target_levels'] = n_target_levels
        if isinstance(targets, (pd.DataFrame, pd.Series)):
            # Make local copy
            targets = targets.copy()
            # Convert Series to DataFrame
            if isinstance(targets, pd.Series):
                targets = targets.to_frame()
            targets.reset_index(drop=True, inplace=True)
            # Infer target name
            target_name = list(targets.columns)[0]

        else:
            if target_name is None:
                target_name = 'target'
            # Convert to a DataFrame
            targets = pd.DataFrame(targets, columns=[target_name])

        # Store name of target
        info['target_name'] = target_name

        # Process target variable
        target_data_type = dict()
        # Get description of X type
        dtype = targets.dtypes[target_name]
        if dtype.name == 'category':  # == pd.Categorical
            encoding = 'ordinal'
            if dtype.ordered:
                data_type = 'ordinal'
            else:
                data_type = 'nominal'
            categories = np.array(targets[target_name].cat.categories)

        elif dtype.name == 'object':  # Most likely a string
            encoding = 'ordinal'
            data_type = 'nominal'
            categories = np.unique(targets[target_name])

        else:
            encoding = None
            data_type = 'numerical'
            categories = np.unique(targets[target_name])

        n_levels = categories.shape[0]

        # Store information
        target_data_type[target_name] = dict(data_type=data_type,
                                             categories=categories,
                                             n_levels=n_levels,
                                             encoding=encoding)
        info['target_levels'] = categories

        # Convert targets
        if target_data_type[target_name]['encoding'] is not None:
            # Retrieve categories and make look-up table to convert categories
            # to numbers
            categories = target_data_type[target_name]['categories']
            categories_lookup = dict({i: j for i, j in zip(categories, np.arange(categories.shape[0]))})
            targets[target_name] = targets[target_name].replace(categories_lookup)

        # Store information on categorical variables
        info['target_data_type'] = target_data_type

        # Store all information and converted targets
        self.targets = targets.values.ravel()
        self.info.update(info)
