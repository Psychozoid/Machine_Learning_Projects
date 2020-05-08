# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset1 = pd.read_excel('Well_B_feat.xls')
dataset2 = pd.read_excel('Well_C_feat.xls')
dataset3 = pd.read_excel('Well_A_feat.xls')
data = dataset1.append([dataset2])
X = data.iloc[:-1, 4:13].values
y = data.iloc[:-1, 13].values
X1 = dataset3.iloc[:-1, 4:13].values
y1 = dataset3.iloc[:-1, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X1 = sc.transform(X1)

"""#Applying Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

#Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
X1 = lda.transform(X1)

#Applying Kernal PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)
X1 = kpca.transform(X1)"""

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Fitting XGBoost Classifier
import xgboost
classifier1 = xgboost.XGBClassifier()
classifier1.fit(X_train, y_train)

#Importing the required libraries
import keras 
from keras.models import Sequential
from keras.layers import Dense

#Fitting the ANN
classifier2 = Sequential()
classifier2.add(Dense(activation = 'relu', input_dim = 9, units = 14, kernel_initializer = 'uniform'))
classifier2.add(Dense(activation = 'relu', units = 14, kernel_initializer = "uniform"))
classifier2.add(Dense(activation = "softmax", units = 5, kernel_initializer = "uniform"))
classifier2.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
classifier2.fit(X_train, y_train, batch_size = 10, epochs = 100)


"""#Using Grid-Search for optimal parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [200, 400, 100, 1000]}
              ]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train) 
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# Applying K-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print('10-fold cross validation mean accuracy Random Forest= ', accuracies.mean())
accuracies1 = cross_val_score(estimator = classifier1, X = X_train, y = y_train, cv = 10)
print('10-fold cross validation mean accuracy XGBoost= ', accuracies1.mean())
accuracies2 = cross_val_score(estimator = classifier2, X = X_train, y = y_train, cv = 10)
print('10-fold cross validation  mean accuracy SVM = ', accuracies2.mean())"""

#Classifying the data using the trained model
y_pred = classifier.predict(X_test)
y_pred1 = classifier.predict(X_train)
y_pred2 = classifier.predict(X1)

y_pred3 = classifier1.predict(X_test)
y_pred4 = classifier1.predict(X_train)
y_pred5 = classifier1.predict(X1)

y_pred6 = classifier2.predict(X_test)
y_pred6 = np.argmax(y_pred6, axis = 1)
y_pred7 = classifier2.predict(X_train)
y_pred7 = np.argmax(y_pred7, axis = 1)
y_pred8 = classifier2.predict(X1)
y_8 = y_pred8
y_pred8 = np.argmax(y_pred8, axis = 1)

depth = dataset3.iloc[:-1, 0].values
dt = dataset3.iloc[:-1, 1].values
dt = np.interp(dt, (dt.min(), dt.max()), (0, 2))
dt_sum = 1*y_8[:, 1] + 2*y_8[:, 2] + 3*y_8[:, 3] + 4*y_8[:, 4]
dt_sum = np.interp(dt_sum, (dt_sum.min(), dt_sum.max()), (0, 2))
plt.plot(dt, depth, color = 'red', label = 'True values')
plt.plot(dt_sum, depth, color = 'blue', label = 'Predicted values') 
plt.xlabel('DT (m/s)')
plt.ylabel('Depth (m)')
plt.title('B&C Test on A')
plt.show()

gr = dataset3.iloc[:-1, 2].values
gr = np.interp(gr, (gr.min(), gr.max()), (0, 2))
gr_sum = 1*y_8[:, 1] + 2*y_8[:, 2] + 3*y_8[:, 3] + 4*y_8[:, 4]
gr_sum = np.interp(gr_sum, (gr_sum.min(), gr_sum.max()), (0, 2))
plt.plot(gr, depth, color = 'red', label = 'True values')
plt.plot(gr_sum, depth, color = 'blue', label = 'Predicted values') 
plt.xlabel('GR (API)')
plt.ylabel('Depth (m)')
plt.title('B&C Test on A')
plt.show()

Np = dataset3.iloc[:-1, 3].values
Np = np.interp(Np, (Np.min(), Np.max()), (0, 2))
Np_sum = 1*y_8[:, 1] + 2*y_8[:, 2] + 3*y_8[:, 3] + 4*y_8[:, 4]
Np_sum = np.interp(Np_sum, (Np_sum.min(), Np_sum.max()), (0, 2))
plt.plot(Np, depth, color = 'red', label = 'True values')
plt.plot(Np_sum, depth, color = 'blue', label = 'Predicted values') 
plt.xlabel('Np (%)')
plt.ylabel('Depth (m)')
plt.title('B&C Test on A')
plt.show()


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm1 = confusion_matrix(y_train, y_pred1)
cm2 = confusion_matrix(y1, y_pred2)
cm3 = confusion_matrix(y_test, y_pred3)
cm4 = confusion_matrix(y_train, y_pred4)
cm5 = confusion_matrix(y1, y_pred5)
cm6 = confusion_matrix(y_test, y_pred6)
cm7 = confusion_matrix(y_train, y_pred7)
cm8 = confusion_matrix(y1, y_pred8)

trace0 = cm.trace()
sum0 = cm.sum()
trace1 = cm1.trace()
sum1 = cm1.sum()
trace2 = cm2.trace()
sum2 = cm2.sum()
trace3 = cm3.trace()
sum3 = cm3.sum()
trace4 = cm4.trace()
sum4 = cm4.sum()
trace5 = cm5.trace()
sum5 = cm5.sum()
trace6 = cm6.trace()
sum6 = cm6.sum()
trace7 = cm7.trace()
sum7 = cm7.sum()
trace8 = cm8.trace()
sum8 = cm8.sum()

print(" Test Set Accuracy of Well - A & B(Random Forest) =", trace0/sum0*100)
#print(" Training Set Accuracy of Well - A & B(istelf)(Random Forest) =", trace1/sum1*100)
print(" Accuracy of Well - C(Random Forest)= ", trace2/sum2*100)
print(" Test Set Accuracy of Well - A & B(XGBoost) =", trace3/sum3*100)
#print(" Training Set Accuracy of Well - A & B(istelf)(XGBoost) =", trace4/sum4*100)
print(" Accuracy of Well - C(XGBoost)= ", trace5/sum5*100)
print(" Test Set Accuracy of Well - A & B(ANN) =", trace6/sum6*100)
#print(" Training Set Accuracy of Well - A(istelf)(ANN) =", trace7/sum7*100)
print(" Accuracy of Well - C(ANN)= ", trace8/sum8*100)