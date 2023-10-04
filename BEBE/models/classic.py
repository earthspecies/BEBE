import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from BEBE.models.supervised_classic_utils import ClassicBehaviorModel

class RandomForest(ClassicBehaviorModel):
  def __init__(self, config):
    super().__init__(config)
    # TODO What is this n_samples_window vs. temporal_window_samples in supervised_classic_utils
    self.n_samples_window = max(int(np.ceil(self.model_config['context_window_sec'] * self.metadata['sr'])), 3)
    self.min_samples_split = self.model_config['min_samples_split']
    self.max_samples = self.model_config['max_samples']
    self.n_jobs = self.model_config['n_jobs']
    self.n_estimators = self.model_config['n_estimators']    
    self.model = RandomForestClassifier(n_estimators=self.n_estimators, min_samples_split=self.min_samples_split, n_jobs=self.n_jobs, verbose=2, max_samples=self.max_samples, random_state = self.config['seed'])

class DecisionTree(ClassicBehaviorModel):
  def __init__(self, config):
    super().__init__(config)
    self.n_samples_window = max(int(np.ceil(self.model_config['context_window_sec'] * self.metadata['sr'])), 3)
    self.min_samples_split = self.model_config['min_samples_split']
    # TODO: Pull relevant parameters from self.model_config that we want to use in hyperparameter search
    # TODO: should class_weight depend on dataset proportions?
    self.model = DecisionTreeClassifier(max_depth=None, min_samples_split=self.min_samples_split, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=self.config['seed'], max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0)

class SupportVectorMachine(ClassicBehaviorModel):
  def __init__(self, config):
    super().__init__(config)
    self.n_samples_window = max(int(np.ceil(self.model_config['context_window_sec'] * self.metadata['sr'])), 3)
    self.min_samples_split = self.model_config['min_samples_split']
    # TODO: Pull relevant parameters from self.model_config that we want to use in hyperparameter search
    # TODO: should class_weight depend on dataset proportions?
    # TODO: LinearSVC is recommended for large datasets
    self.model = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=self.config['seed'], max_iter=1000)