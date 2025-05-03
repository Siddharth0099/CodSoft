import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from imblearn.over_sampling import SMOTE

data = pd.read_csv("creditcard1.csv")

print(data.head())
print(data.describe())

fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlier_fraction = len(fraud) / float(len(valid))
print(f"Outlier Fraction: {outlier_fraction:.6f}")
print(f"Fraud Cases: {len(fraud)}")
print(f"Valid Transactions: {len(valid)}")

print(fraud['Amount'].describe())
print(valid['Amount'].describe())

corrmat = data.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

X = data.drop(['Class'], axis=1)
Y = data['Class']
xData = X.values
yData = Y.values

xTrain, xTest, yTrain, yTest = train_test_split(
    xData, yData, test_size=0.2, random_state=42, stratify=Y)

print("Class distribution in yTrain:", np.bincount(yTrain))
print("Class distribution in yTest:", np.bincount(yTest))

if len(np.unique(yTrain)) == 1:
    print("Error: Training set contains only one class. Please check data splitting.")
else:
    smote = SMOTE(random_state=42)
    xTrain_resampled, yTrain_resampled = smote.fit_resample(xTrain, yTrain)

    rfc = RandomForestClassifier(class_weight='balanced', random_state=42)
    rfc.fit(xTrain_resampled, yTrain_resampled)
    yPred = rfc.predict(xTest)

    print("NaN values in yTest before imputation:", np.isnan(yTest).sum())
    imputer = SimpleImputer(strategy='most_frequent')
    yTest = imputer.fit_transform(yTest.reshape(-1, 1)).ravel()
    print("NaN values in yTest after imputation:", np.isnan(yTest).sum())

    accuracy = accuracy_score(yTest, yPred)
    precision = precision_score(yTest, yPred, zero_division=0)
    recall = recall_score(yTest, yPred, zero_division=0)
    f1 = f1_score(yTest, yPred, zero_division=0)
    mcc = matthews_corrcoef(yTest, yPred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("MCC:", mcc)

    print("Unique values in yPred:", np.unique(yPred))
    print("Unique values in yTest:", np.unique(yTest))

    if len(np.unique(yPred)) > 1 and len(np.unique(yTest)) > 1:
        conf_matrix = confusion_matrix(yTest, yPred, labels=[0, 1])
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.tight_layout()
        plt.show()
