from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial import distance
import numpy as np


class ISOFAEnvironment:
    def __init__(self, core_table, target, max_try_num):
        self.data = core_table
        self.current_training_set = None
        self.target = target
        self.max_try_num = max_try_num
        self.try_num = 0
        self.y = self.data[target]
        self.data.drop([target], axis=1,inplace=True)

        self.selected_features = []  # Indices of selected features
        self.current_model = None
        self.cur_score = 0
        self.original_score = 0
        self.cur_state = None
        self.prev_score = 0
        self.state_len = len(self.data.columns)
        self.initial_feature_fraction = 0.3

        self.init_environment()

    def init_environment(self):
        print('-' * 20 + "Init:" + '-' * 20)
        available_features = self.data.columns
        num_features_to_select = int(len(available_features) * self.initial_feature_fraction)
        selected_features = np.random.choice(available_features, num_features_to_select, replace=False)
        self.selected_features = selected_features.tolist()  # Store feature names directly

        # Create the training set using only the selected features plus the target
        self.current_training_set = self.data[list(selected_features)]

        X_train, X_test, y_train, y_test = self.split_data(self.current_training_set)
        self.current_model = self.train_subsequent_learner(X_train, y_train)
        self.cur_score = self.test_subsequent_learner(X_test, y_test)
        self.original_score = self.cur_score
        self.update_state()

    def reset(self):
        self.selected_features = []
        self.try_num = 0
        self.init_environment()

    def update_state(self):
        # State vector indicating if each feature is selected
        self.cur_state = [1 if i in self.selected_features else 0 for i in
                          range(len(self.data.columns))]

    def step(self, action):
        """
        Executes an action, which is the selection of a feature, and updates the model and scores.
        :param action: The index of the feature to select.
        :return: next_state, reward, done - Returns the new state, the reward obtained, and a flag indicating if the episode is done.
        """
        feature_name = self.data.columns[action]
        if feature_name in self.selected_features:
            # If the feature is already selected, return the current state and no reward.
            done = len(self.selected_features) == len(self.data.columns)

            return self.cur_state, 0, done

        self.selected_features.append(feature_name)

        # Update the model using only the selected features.
        selected_columns = self.selected_features
        self.current_training_set = self.data[selected_columns]

        X_train, X_test, y_train, y_test = self.split_data(self.current_training_set)

        # Re-train the model with the newly selected features.
        self.current_model = self.train_subsequent_learner(X_train, y_train)
        new_score, jsd = self.test_subsequent_learner(X_test, y_test)

        # Calculate the reward as the improvement in score from the previous step.
        reward = new_score - self.prev_score - jsd
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
        # Predict probabilities for the positive class
        y_pred_prob = self.current_model.predict_proba(X_test)[:, 1]

        # Convert predictions to a histogram/distribution if using a discrete representation
        # Here, you could simply use y_pred_prob directly if treating them as a continuous distribution
        pred_hist, _ = np.histogram(y_pred_prob, bins=2, range=(0,1), density=True)
        pred_hist = pred_hist / np.sum(pred_hist)  # Normalize to form a probability distribution

        # Convert true labels to a histogram/distribution
        true_hist, _ = np.histogram(y_test, bins=2, range=(0,1), density=True)
        true_hist = true_hist / np.sum(true_hist)  # Normalize

        # Compute Jensen-Shannon Divergence
        jsd = self.compute_jsd(pred_hist, true_hist)
        print("Jensen-Shannon Divergence:", jsd)

        # Compute AUC for comparison
        auc = roc_auc_score(y_test, y_pred_prob)
        print("AUC:", auc)

        return auc, jsd


    def get_training_dataset(self):
        return self.split_data(self.current_training_set.loc[:, self.selected_features + [self.target]])

    def get_current_features(self):
        return [self.current_training_set.columns[i] for i in self.selected_features]

    def get_state_len(self):
        # Returns the length of the state vector, which corresponds to the number of features
        return len(self.current_training_set.columns)

    def compute_jsd(self, p, q):
        """ Compute the Jensen-Shannon divergence using square of the JSD distance. """
        return distance.jensenshannon(p, q) ** 2