import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

my_model = joblib.load('models/rfc.joblib')
X_test = pd.DataFrame({
	'X': [9.375,6.995,0,9.4,9.4], 
	'Y': [3.0625,5.125,3.0625,3,3],
	'Z': [1.51,0.3875,1.93,1.8,1.3]
	})
y_pred = my_model.predict(X_test)

print("The predicted steps and the respective probabilities are:\n",y_pred)