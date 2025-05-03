import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']
df = pd.read_csv('irisdata.data', names=columns)
print("First five rows of the dataset:")
print(df.head())

print("\nDataset statistics:")
print(df.describe())

sns.pairplot(df, hue='Class_labels')
plt.show()

data = df.values
X = data[:, 0:4]
Y = data[:, 4]

Y_Data = np.array([
    np.average(X[:, i][Y == j].astype('float32'))
    for i in range(X.shape[1])
    for j in np.unique(Y)
])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)

X_axis = np.arange(len(columns) - 1)
width = 0.25

plt.bar(X_axis, Y_Data_reshaped[0], width, label='Setosa')
plt.bar(X_axis + width, Y_Data_reshaped[1], width, label='Versicolour')
plt.bar(X_axis + width * 2, Y_Data_reshaped[2], width, label='Virginica')
plt.xticks(X_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3, 1))
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, y_train)

predictions = svn.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report
print("\nAccuracy of SVM model:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

X_new = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])
prediction = svn.predict(X_new)
print("\nPrediction of Species:", prediction)

import pickle
with open('SVM.pickle', 'wb') as f:
    pickle.dump(svn, f)
print("\nModel saved as 'SVM.pickle'.")

with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)

loaded_model_prediction = model.predict(X_new)
print("Prediction using loaded model:", loaded_model_prediction)
