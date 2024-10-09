# imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import log_loss, precision_score, f1_score, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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
print(X_train.corr(method='pearson'))


''' step 4 '''


# Support Vector Machine

params_grid_svc = {
    'C': [1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto']
}
grid_search_svc = GridSearchCV(SVC(probability=True), params_grid_svc, cv=5, scoring='neg_log_loss', n_jobs=-1)
grid_search_svc.fit(X_train, y_train)
my_svc = grid_search_svc.best_estimator_
best_params_svc = grid_search_svc.best_params_
print("\nBest parameters for Support Vector Classifier:")
print(best_params_svc)

# Evaluation of SVC

y_pred_svc = my_svc.predict(X_test)
y_pred_probability_svc = my_svc.predict_proba(X_test)
cross_entropy_svc = log_loss(y_test, y_pred_probability_svc)
print("\nThe Cross Entropy for the Support Vector Classifier after grid search is:")
print(cross_entropy_svc)

# Decision Tree Classifier

params_grid_dtc = {
    'criterion': ['gini','entropy'],
    'max_depth': [5, 10, 25, 100],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [2,3,4],
    'max_features': [5,10]
}
grid_search_dtc = GridSearchCV(DecisionTreeClassifier(), params_grid_dtc, cv=5, scoring='neg_log_loss', n_jobs=-1)
grid_search_dtc.fit(X_train, y_train)
my_dtc = grid_search_dtc.best_estimator_
best_params_dtc = grid_search_dtc.best_params_
print("\nBest parameters for Decision Tree Classifier:")
print(best_params_dtc)

# Evaluation of DTC

y_pred_dtc = my_dtc.predict(X_test)
y_pred_probability_dtc = my_dtc.predict_proba(X_test)
cross_entropy_dtc = log_loss(y_test, y_pred_probability_dtc)
print("\nThe Cross Entropy for the Decision Tree Classifier is:")
print(cross_entropy_dtc)

# Random Forest Classifier with Grid Search

params_grid_rfc = {
    'n_estimators': [5],
    'criterion': ['gini','entropy'],
    'max_depth': [1, 3, 5, 10],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [2,3,4],
    'max_features': [5,10]
}
grid_search_rfc = GridSearchCV(RandomForestClassifier(), params_grid_rfc, cv=5, scoring='neg_log_loss', n_jobs=-1)
grid_search_rfc.fit(X_train, y_train)
my_rfc = grid_search_rfc.best_estimator_
best_params_rfc = grid_search_rfc.best_params_
print("\nBest parameters for Random Forest Classifier using Grid Search:")
print(best_params_rfc)

# Evaluation of RFC with Grid Search

y_pred_rfc = my_rfc.predict(X_test)
y_pred_probability_rfc = my_rfc.predict_proba(X_test)
cross_entropy_rfc = log_loss(y_test, y_pred_probability_rfc)
print("\nThe Cross Entropy for the Random Forest Classifier using Grid Search is:")
print(cross_entropy_rfc)

# Random Forest Classifier with Random Search

grid_search_rfc_rand = RandomizedSearchCV(RandomForestClassifier(), params_grid_rfc, n_iter=5, scoring='neg_log_loss',random_state=50, n_jobs=-1)
grid_search_rfc_rand.fit(X_train, y_train)

my_rfc_rand = grid_search_rfc_rand.best_estimator_
best_params_rfc_rand = grid_search_rfc_rand.best_params_
print("\nBest parameters for Random Forest Classifier using Randomized Search:")
print(best_params_rfc_rand)

# Evaluation of RFC with Randomized Search

y_pred_rfc_rand = my_rfc_rand.predict(X_test)
y_pred_probability_rfc_rand = my_rfc_rand.predict_proba(X_test)
cross_entropy_rfc_rand = log_loss(y_test, y_pred_probability_rfc_rand)
print("\nThe Cross Entropy for the Random Forest Classifier using Randomized Search is:")
print(cross_entropy_rfc_rand)


''' step 5 '''


# Precision, Accuracy and f1 Scores for Each Model

precision_svc = precision_score(y_test, y_pred_svc, average='macro')
accuracy_svc = accuracy_score(y_test, y_pred_svc)
f1_svc = f1_score(y_test, y_pred_svc, average='macro')

print("\nFor the Support Vector Classifier:\nThe precision is ",precision_svc,"\nThe accuracy is ",accuracy_svc,"\nThe f1 score is ",f1_svc)

precision_dtc = precision_score(y_test, y_pred_dtc, average='macro')
accuracy_dtc = accuracy_score(y_test, y_pred_dtc)
f1_dtc = f1_score(y_test, y_pred_dtc, average='macro')

print("\nFor the Decision Tree Classifier:\nThe precision is ",precision_dtc,"\nThe accuracy is ",accuracy_dtc,"\nThe f1 score is ",f1_dtc)

precision_rfc = precision_score(y_test, y_pred_rfc, average='macro')
accuracy_rfc = accuracy_score(y_test, y_pred_rfc)
f1_rfc = f1_score(y_test, y_pred_rfc, average='macro')

print("\nFor the Random Forest Classifier:\nThe precision is ",precision_rfc,"\nThe accuracy is ",accuracy_rfc,"\nThe f1 score is ",f1_rfc)

precision_rfc_rand  = precision_score(y_test, y_pred_rfc_rand, average='macro')
accuracy_rfc_rand = accuracy_score(y_test, y_pred_rfc_rand)
f1_rfc_rand = f1_score(y_test, y_pred_rfc_rand, average='macro')

print("\nFor the Random Forest Classifier with the Randomized Search:\nThe precision is ",precision_rfc_rand,"\nThe accuracy is ",accuracy_rfc_rand,"\nThe f1 score is ",f1_rfc_rand)

# Confusion Matrix for Each Model