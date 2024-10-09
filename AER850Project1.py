# imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier

''' step 1 '''

# data processing

df = pd.read_csv('data/Project_1_Data.csv')

''' step 2 '''

# in order to avoid data leaks during visualization, the train-test split is done now

my_splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 777)
for train_index, test_index in my_splitter.split(df, df["Step"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)

X_train = strat_df_train.drop("Step", axis = 1)
y_train = strat_df_train["Step"]
X_test = strat_df_test.drop("Step", axis = 1)
y_test = strat_df_test["Step"]

# data visualization

fig = plt.figure()

ax = fig.add_subplot(projection='3d')
scatter_plot=  ax.scatter(X_train['X'],X_train['Y'],X_train['Z'],c=y_train,cmap='winter_r')
# cbar = plt.colorbar(scatter_plot)
# cbar.set_label("Step")

plt.show()

''' step 3 '''

sns.heatmap(np.abs(X_train.corr()))
plt.show()
print("\nThe Correlation Matrix for the three dependent variables is as follows:\n")
print(X_train.corr())

''' step 4 '''

# Support Vector Machine

params_grid_svc = {
    'C': [1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto']
}
grid_search_svc = GridSearchCV(SVC(probability=True), params_grid_svc, cv=5, scoring='neg_log_loss')
grid_search_svc.fit(X_train, y_train)
my_svc = grid_search_svc.best_estimator_
best_params = grid_search_svc.best_params_
print("\nBest parameters:")
print(best_params)

# Cross Entropy for SVC

y_pred_probability = my_svc.predict_proba(X_test)
cross_entropy_svc = log_loss(y_test, y_pred_probability)
print("\nThe Cross Entropy for the Support Vector Classifier after grid search is:")
print(cross_entropy_svc)

# Decision Tree

