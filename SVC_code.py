#importing modules
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


"""--------------------------------------MODEL TRAINING--------------------------------------"""

data = pd.read_csv('synthetic_diabetes_data.csv')
df = pd.DataFrame(data)
df['diabetes'].value_counts()

x_train, x_test, y_train, y_test = train_test_split(df.drop('diabetes', axis=1), df['diabetes'],
test_size=0.2, random_state=42)

#handling data imbalance using SMOTE
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)
y_train.value_counts()

#Feature Scaling for numerical features
features = ['age','bmi']
scaler = StandardScaler()
x_train_numerical_scaled= scaler.fit_transform(x_train[features])
x_test_numerical_scaled= scaler.transform(x_test[features])
#combine x_train with x_train_numerical_scaled and x_test with x_test_numerical_scaled
x_train = pd.DataFrame(x_train_numerical_scaled, columns=features)
x_test = pd.DataFrame(x_test_numerical_scaled, columns=features)

#svc model
params = {
    'C': [0.1, 1, 10,15,18,20],
    'gamma': ['scale', 0.01, 0.1, 1,1.54,2,2.5,3]
}
grid = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), params, cv=5)       #cv=5 is like k in k-fold cross-validation
#above line applies grid search with cross-validation to find the best hyperparameters for the SVC model
grid.fit(x_train, y_train) #this line trains the SVC model for the whole training data at once
print(grid.best_params_)
y_pred = grid.predict(x_test)

"""--------------------------------------MODEL EVALUATION--------------------------------------"""

cf = confusion_matrix(y_test, grid.predict(x_test))
tn, fp, fn, tp = cf.ravel()

accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*precision*recall/(precision+recall)
print("Accuracy: ",accuracy)
print("Precision: ",precision)
print("Recall: ",recall)
print("F1 Score: ",f1_score)
print(tn,fp,fn,tp)
#visualization
sns.heatmap(cf, annot=True, cmap='Blues',fmt='d')
#plt.xlabel('Predicted')
#plt.ylabel('Actual')
plt.show()