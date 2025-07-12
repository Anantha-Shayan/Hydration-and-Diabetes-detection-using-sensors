import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

"""train_test_split and class balancing"""
df['Outcome'].value_counts()  #class imbalance
x_train, x_test, y_train, y_test = train_test_split(df.drop('Outcome', axis=1), df['Outcome'], test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)
y_train.value_counts() #balanced classes
features = ['Age','BMI']

"""Feature Scaling for numerical features"""
scaler = StandardScaler()
x_train_numerical_scaled= scaler.fit_transform(x_train[features])
x_test_numerical_scaled= scaler.transform(x_test[features])
x_train = pd.DataFrame(x_train_numerical_scaled, columns=features)
x_test = pd.DataFrame(x_test_numerical_scaled, columns=features)

"""Model selection"""
models = {
    'SVM': SVC(C=10, kernel='rbf', class_weight='balanced'),
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    ' XGBoost': xgb.XGBClassifier(scale_pos_weight=1, random_state=42)
}

# === 4. Evaluate Each Model ===
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", acc)

"""Hyperparameter tuning for SVM"""
params = {
    'C': [0.1, 1, 10,15,18,20],
    'gamma': ['scale', 0.01, 0.1, 1,1.54,2,2.5,3]
}
grid = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), params, cv=5)
grid.fit(x_train, y_train)
print(grid.best_params_)
y_pred = grid.predict(x_test)

"""Model Evaluation"""
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
sns.heatmap(cf, annot=True, cmap='Blues',fmt='d')
plt.show()
