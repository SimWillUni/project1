# imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

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

fig, axes = plt.subplots(2, 2)

axes[0, 0].hist(y_train, bins=12)
axes[0, 0].set_title('Step Values')
axes[0, 1].hist(X_train['X'])
axes[0, 1].set_title('X Values')
axes[1, 0].hist(X_train['Y'])
axes[1, 0].set_title('Y Values')
axes[1, 1].hist(X_train['Z'])
axes[1, 1].set_title('Z Values')

plt.tight_layout()
plt.show()

''' step 3 '''

