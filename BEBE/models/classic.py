import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from BEBE.models.supervised_classic_utils import ClassicBehaviorModel

class RandomForest(ClassicBehaviorModel):
  def __init__(self, config):
    super().__init__(config)
    self.min_samples_split = self.model_config['min_samples_split']
    self.max_samples = self.model_config['max_samples']
    self.n_jobs = self.model_config['n_jobs']
    self.n_estimators = self.model_config['n_estimators']
    self.class_weight =  self.model_config['class_weight']
    self.model = RandomForestClassifier(n_estimators=self.n_estimators, min_samples_split=self.min_samples_split, n_jobs=self.n_jobs, class_weight= self.class_weight, verbose=2, max_samples=self.max_samples, random_state = self.config['seed'])

class DecisionTree(ClassicBehaviorModel):
  def __init__(self, config):
    super().__init__(config)
    self.min_samples_split = self.model_config['min_samples_split']
    self.class_weight =  self.model_config['class_weight']
    self.model = DecisionTreeClassifier(min_samples_split=self.min_samples_split, random_state=self.config['seed'], class_weight= self.class_weight)

class SupportVectorMachine(ClassicBehaviorModel):
  def __init__(self, config):
    super().__init__(config)
    self.min_samples_split = self.model_config['min_samples_split']
    self.max_iter = self.model_config['max_iter']
    self.C = self.model_config['regularization_parameter']
    self.intercept_scaling = self.model_config['intercept_scaling']
    self.class_weight =  self.model_config['class_weight']
    self.model = LinearSVC(dual=False, C=self.C, intercept_scaling=self.intercept_scaling, class_weight= self.class_weight, verbose=2, random_state=self.config['seed'], max_iter=self.max_iter)