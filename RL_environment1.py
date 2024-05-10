import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial import distance
import numpy as np
from sklearn.metrics import mutual_info_score

import scipy.stats as ss


# TODO: - Modify action space: Add Feature/Remove Feature
#       - Add state variables: Information Gain
#       - Modify algorithm to select subset of features, not only train once per feature
#


class ISOFAEnvironment:
    """
    State = [[Features: 0,1 given n features], [Information Gain: given n features with respect to the target variable]]
    Reward = S_t+1 AUC - S_t AUC - S_t+1 JSD
    Action = [Feature name]

    """

    def __init__(self, core_table, target, max_try_num):
        self.temp = core_table.copy()
        self.data = core_table
        self.current_training_set = None
        self.target = target
        self.max_try_num = max_try_num
        self.try_num = 0
        self.y = self.data[target]
        self.data.drop([target], axis=1, inplace=True)
        self.action_space = {'add': [], 'delete': []}
        self.selected_features = []  # Indices of selected features
        self.current_model = None
        self.cur_score = 0
        self.original_score = 0
        self.cur_state = None
        self.prev_score = 0
        self.state_len = len(self.data.columns)
        self.initial_feature_fraction = 0.2
        self.original_flag = True

        self.init_environment()

    def init_environment(self):
        print('-' * 20 + "Init:" + '-' * 20)
        available_features = self.data.columns
        num_features_to_select = 1
        selected_features = np.random.choice(available_features, num_features_to_select, replace=False)
        self.selected_features = selected_features.tolist()  # Store feature names directly

        # Create the training set using only the selected features plus the target
        self.current_training_set = self.data[list(selected_features)]

        X_train, X_test, y_train, y_test = self.split_data(self.current_training_set)
        self.current_model = self.train_subsequent_learner(X_train, y_train)
        self.cur_score, jsd = self.test_subsequent_learner(X_test, y_test)

        if self.original_flag:
            self.original_score = self.cur_score
            self.original_flag = False

        self.update_state()

    def reset(self):
        self.selected_features = []
        self.try_num = 0
        self.init_environment()

    def update_state(self):
        # State vector indicating if each feature is selected

        self.cur_state = [[1 if i in self.selected_features else 0 for i in
                           self.data.columns],
                          [self.calc_mutual_info(feat) for feat in self.data.columns],
                          [self.calc_correlation(feat) for feat in self.data.columns]
                          ]

    def step(self, action):
        """
        Executes an action, which is the selection of a feature, and updates the model and scores.
        :param action: The index of the feature to select.
        :return: next_state, reward, done - Returns the new state, the reward obtained, and a flag indicating if the episode is done.
        """
        feature_name = self.data.columns[action]


        feature_name = self.data.columns[action]
        if feature_name in self.selected_features:
            # If the feature is already selected, return the current state and no reward.
            done = len(self.selected_features) == len(self.data.columns)

            return self.cur_state, 0, done

        # Update the model using only the selected features.
        self.selected_features.append(feature_name)
        selected_columns = self.selected_features
        self.current_training_set = self.data[selected_columns]

        X_train, X_test, y_train, y_test = self.split_data(self.current_training_set)

        # Re-train the model with the newly selected features.
        self.current_model = self.train_subsequent_learner(X_train, y_train)
        new_score, jsd = self.test_subsequent_learner(X_test, y_test)

        # Calculate the reward as the improvement in score from the previous step.
        reward = new_score - self.prev_score

        self.prev_score = self.cur_score
        self.cur_score = new_score

        # Update the environment's state to reflect the new set of selected features.
        self.update_state()

        # Check if the maximum number of trials has been reached.
        done = self.try_num >= self.max_try_num or len(self.selected_features) == len(self.data.columns)

        self.try_num += 1

        return self.cur_state, reward, done

    def split_data(self, dataset):
        return train_test_split(dataset, self.y, test_size=0.30, random_state=42)

    def train_subsequent_learner(self, X_train, y_train):
        model = XGBClassifier(eval_metric='auc', enable_categorical=True,
                              use_label_encoder=False)
        model.fit(X_train, y_train)
        return model

    def test_subsequent_learner(self, X_test, y_test):
        # Predict probabilities for each class
        y_pred_prob = self.current_model.predict_proba(X_test)

        # Convert true labels into one-hot encoding
        encoder = OneHotEncoder()
        y_true_one_hot = encoder.fit_transform(
            y_test.values.reshape(-1, 1)).toarray()  # Convert to dense array if necessary
        y_pred = self.current_model.predict(X_test)

        # Compute Jensen-Shannon Divergence
        jsd = self.jsd(y_pred_prob, y_true_one_hot)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        # Compute AUC for comparison
        auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovo')

        print("AUC:", auc)

        return accuracy, jsd

    def get_training_dataset(self):
        return self.split_data(self.current_training_set.loc[:, self.selected_features + [self.target]])

    def get_current_features(self):
        return [self.current_training_set.columns[i] for i in self.selected_features]

    def get_state_len(self):
        # Returns the length of the state vector, which corresponds to the number of features
        return len(self.current_training_set.columns)

    def jsd(self, predictions, actuals):
        # Assuming `predictions` is a list of lists, each inner list is a probability distribution for a data point
        # Assuming `actuals` is a list of class indices, one-hot encoded
        from scipy.spatial.distance import jensenshannon
        import numpy as np

        jsd_values = [jensenshannon(pred, act) ** 2 for pred, act in zip(predictions, actuals)]
        return np.mean(jsd_values)

    def calc_mutual_info(self, feat):
        if self.data[feat].dtype in ['int64', 'float64']:
            # Discretize the column
            discretized = pd.cut(self.data[feat], bins=10, labels=False, duplicates='drop')
            # discretized = discretized.fillna(-1)  # TEMP STRATEGY FOR NAN VALUES
            mi = mutual_info_score(discretized, self.y)
        else:
            self.data[feat].cat.add_categories("Unknown")
            # filled_series = self.data[feat].fillna('Unknown')  # TEMP STRATEGY FOR NAN VALUES
            # print(self.df_sketch[feat], self.df_sketch[target])
            mi = mutual_info_score(self.data[feat], self.y)

        return mi

    def calc_correlation(self, feat):
        if self.data[feat].dtype in ['int64', 'float64']:
            ab = self.temp[[feat, self.target]]
            corr = ab.corr().iloc[0, 1]
        else:
            self.data[feat].cat.add_categories("Unknown")
            # filled_series = self.data[feat].fillna('Unknown')  # TEMP STRATEGY FOR NAN VALUES
            corr = self.cramers_v(self.data[feat], self.y)

        return 0 if np.isnan(corr) else corr

    def cramers_v(self, x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
