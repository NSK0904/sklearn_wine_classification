#!/usr/bin/env python
# coding: utf-8

# # Loading Wine Recognition Dataset

# Firstly I will import datasets from sklearn.

# In[1]:


from sklearn import datasets


# Then from the sklearn datasets I will load the 'wine' dataset and get its full description.

# In[2]:


wine_data = datasets.load_wine()
print(wine_data['DESCR'])


# # Visualize and Preprocess Data

# I will now check the shapes of the 'data' and target' fields

# In[3]:


X = wine_data['data']
y = wine_data['target']

print('data.shape\t',X.shape,
      '\ntarget.shape \t',y.shape)


# With this it is confirmed that there are 178 samples (rows) and 13 feaures (columns)
# I will now build a pandas DataFrame, to hold the data so that we can visualize the dataset into a tabular form.

# In[4]:


import numpy as np
import pandas as pd

datawine = pd.DataFrame(data= np.c_[X,y],columns= wine_data['feature_names'] + ['target'])
datawine


# With this DataFrame I can check for any missing values.

# In[5]:


datawine.isnull().sum()


# This confirms that there are no missing values and categorical data
# The final step to data preprocessing is Feature Scaling. This is done by importing StandardScaler from sklearn.preprocessing

# In[6]:


from sklearn.preprocessing import StandardScaler

st_x= StandardScaler()
X= st_x.fit_transform(X)


# # Split Data For Training & Testing

# To train and test our model effectively, we must first separate the data into a training set, which we will feed to our model along with the training labels. The model will then be tested on the 'test' data after it has been trained to determine its real-world applicability.
# 
# The train test split() method in Scikit-learn comes in handy here. test size specifies how much data is set aside for testing. We want to train our model on enough data to make good predictions, but we also need enough test data to see if we've overfitted the model. Therefore I will choose to test with 20% of the data. This means 80% of data will used for training.

# In[7]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 #I chose 42 for random_state as is the most common number used for random_state
)

print(len(X_train),' samples in training data\n',
      len(X_test),' samples in test data\n', )


# # Decision Tree

# I will first create a decision tree classifier object and define a parameter grid for parameter tuning.
# For the param_grid argument, I created a dictionary and chose 3 parameters used in the Decision Tree Classifer.
# max-depth for the maximum depth of the tree - default (none)
# min_samples_split for minimum number of samples required to split an internal node - default (2)
# min_samples_leaf for minimum number of samples required to be at a leaf node - default (1)

# In[8]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

dt = DecisionTreeClassifier()

param_grid = {
    'max_depth': [2,3,4,5,6,7],
    'min_samples_split': [2,3,4,5,6,7],
    'min_samples_leaf': [1,2,3,4]
}


# I created a GridSearchCV object and included the estimator, the param_grid and set the K-fold Cross Validation to 5 which is the default for determining the cross-validation splitting strategy. Then I Fit the GridSearchCV object to the training data.

# In[9]:


dt_grid = GridSearchCV(dt, param_grid, cv=5)
dt_grid.fit(X_train, y_train)


# In[10]:


print("Best parameters: ", dt_grid.best_estimator_)
print("Best parameters: ", dt_grid.best_params_)


# As shown above, the Max_depth is chosen as 3. An increasing depth makes a tree model more expressive but a tree too deep will overfit the data, so 3 is good enough for the depth.
# Min_samples_leaf is chosen as 1 which is the default value.
# Min_samples_split is chosen as 2 which is the default value.
# Then the best estimator object is fit to the training data, and the classifier is evaluated.

# In[11]:


from sklearn import metrics

best_dt = dt_grid.best_estimator_
best_dt.fit(X_train, y_train)
y_train_pred_dt = best_dt.predict(X_train)
y_pred_dt = best_dt.predict(X_test)

acc_train = metrics.accuracy_score(y_train, y_train_pred_dt)
print("DT acc_train: %f" %acc_train )
acc = metrics.accuracy_score(y_test, y_pred_dt)
print("DT acc: %f" %acc )


# The accuracy score in % for Decision Tree is 94.4

# In[12]:


recall_train = metrics.recall_score(y_train, y_train_pred_dt, average=None)
print("DT recall_train: ", recall_train )
recall = metrics.recall_score(y_test, y_pred_dt, average=None)
print("DT recall: "  ,recall )


# The recall score is 1 which is the best for class_1 and then followed by class_0 but the score is low for class_2.

# In[13]:


prec_train = metrics.precision_score(y_train, y_train_pred_dt, average=None)
print("DT precision_train:", prec_train )
prec = metrics.precision_score(y_test, y_pred_dt, average=None)
print("DT precision:", prec )


# The precision score is 1 which is the best for class_2 and class_0 and 0.875 for class_1 respectively.

# In[14]:


f1_train = metrics.f1_score(y_train, y_train_pred_dt, average=None)
print("DT f1 train:", f1_train )
f1 = metrics.f1_score(y_test, y_pred_dt, average=None)
print("DT f1:", f1 )


# The F1 score is best in class_0, followed by decent scores in both class_1 and class_2.

# In[15]:


conf_mat = metrics.confusion_matrix(y_test, y_pred_dt)
print("DT conf_mat: %f")
print(conf_mat)


# Based on all the scores a classification report can be generated for the Decision Tree Classification Model.

# In[16]:


print("Decision Tree Classification Report:\n", metrics.classification_report(y_test, y_pred_dt, digits=4))


# # Random Forest

# I will first create a random classifier object and define a parameter grid for parameter tuning.
# For the param_grid argument, I created a dictionary and chose 3 parameters used in the Random Forest Classifer.
# criterion to measure the quality of a split - default (gini)
# max-depth for the maximum depth of the tree - default (none)
# n-estimators for the number of trees in the forest - default (100)

# In[17]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

param_grid1 = {
    'n_estimators': [10,30,50, 100, 150,200,300],
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'criterion': ['gini', 'entropy']
}


# I created a GridSearchCV object and included the estimator, the param_grid and set the K-fold Cross Validation to 5 which is the default for determining the cross-validation splitting strategy. Then I Fit the GridSearchCV object to the training data.

# In[18]:


grid_search = GridSearchCV(clf, param_grid=param_grid1, cv=5)
grid_search.fit(X_train, y_train)


# In[19]:


print("Best parameters: ", grid_search.best_estimator_)
print("Best parameters:", grid_search.best_params_)


# As shown above, the criterion is chosen as entropy which is not the default (Gini).
# Max_depth is chosen as 5. An increasing depth makes a tree model more expressive but a tree too deep will overfit the data, so 5 is good enough for the depth.
# N_estimators is chosen as 100 which is the default value. Increasing the number of trees can improve the performance of the classifier but also increases the computational cost.
# Then the best estimator object is fit to the training data, and the classifier is evaluated.

# In[20]:


best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
y_train_pred_rf = best_rf.predict(X_train)
y_pred_rf = best_rf.predict(X_test)

acc_train = metrics.accuracy_score(y_train, y_train_pred_rf)
print("RF acc_train: %f" %acc_train )
acc = metrics.accuracy_score(y_test, y_pred_rf)
print("RF acc: %f" %acc )


# The accuracy score for Random Forest Classifier in % is 100, which is absolute accuracy

# In[21]:


recall_train = metrics.recall_score(y_train, y_train_pred_rf, average=None)
print("RF recall_train:", recall_train )
recall = metrics.recall_score(y_test, y_pred_rf, average=None)
print("RF recall:", recall )


# The recall scores are best in all classes.

# In[22]:


prec_train = metrics.precision_score(y_train, y_train_pred_rf, average=None)
print("RF precision_train:", prec_train )
prec = metrics.precision_score(y_test, y_pred_rf, average=None)
print("RF precision:", prec )


# The precision scores are best in all classes.

# In[23]:


f1_train = metrics.f1_score(y_train, y_train_pred_rf, average=None)
print("RF f1 train:", f1_train )
f1 = metrics.f1_score(y_test, y_pred_rf, average=None)
print("RF f1:", f1 )


# The F1 scores are best in all classes.

# In[24]:


conf_mat = metrics.confusion_matrix(y_test, y_pred_rf)
print("RF conf_mat:")
print(conf_mat)


# Based on all the scores a classification report can be generated for the Random Forest Classification Model.

# In[25]:


print("Random Forest Classification Report:\n", metrics.classification_report(y_test, y_pred_rf, digits=4))


# # SVM

# I will first create a SVC classifier object and define a parameter grid for parameter tuning.
# For the param_grid argument, I created a dictionary and chose 3 parameters used in the Decision Tree Classifer.
# C is a Regularization parameter - default (1)
# kernel Specifies the kernel type to be used in the algorithm - default (rbf)
# gamma is the Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’ - default(scale)

# In[26]:


from sklearn.svm import SVC

svmc = SVC()

param_grid2 = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}


# I created a GridSearchCV object and included the estimator, the param_grid and set the K-fold Cross Validation to 5 which is the default for determining the cross-validation splitting strategy. Then I Fit the GridSearchCV object to the training data.

# In[27]:


grid_search2 = GridSearchCV(svmc, param_grid=param_grid2, cv=5)
grid_search2.fit(X_train, y_train)


# In[28]:


print("Get parameters: ", grid_search2.best_estimator_)
print("Best parameters:", grid_search2.best_params_)


# As shown above, the C value is chosen as 1 which is the default value.
# Kernel is sigmoid so this function is equivalent to a two-layer, perceptron model of the neural network, which is used as an activation function for artificial neurons.
# Gamma is chosen as scale which is the default value.
# Then the best estimator object is fit to the training data, and the classifier is evaluated.

# In[29]:


best_svc = grid_search2.best_estimator_
best_svc.fit(X_train, y_train)
y_train_pred_svc = best_svc.predict(X_train)
y_pred_svc = best_svc.predict(X_test)

acc_train = metrics.accuracy_score(y_train, y_train_pred_svc)
print("SVC acc_train:", acc_train )
acc = metrics.accuracy_score(y_test, y_pred_svc)
print("SVC acc:", acc )


# The accuracy score for SVC in % is 97.2

# In[30]:


recall_train = metrics.recall_score(y_train, y_train_pred_svc, average=None)
print("SVC recall_train:", recall_train )
recall = metrics.recall_score(y_test, y_pred_svc, average=None)
print("SVC recall:", recall )


# The recall score is best with class_0 and class_2, and decent with class_1.

# In[31]:


prec_train = metrics.precision_score(y_train, y_train_pred_svc, average=None)
print("SVC precision_train:", prec_train )
prec = metrics.precision_score(y_test, y_pred_svc, average=None)
print("SVC precision:", prec )


# The precision score is best with class_0 and class_1, however low with class_2.

# In[32]:


f1_train = metrics.f1_score(y_train, y_train_pred_svc, average=None)
print("SVC f1 train:", f1_train )
f1 = metrics.f1_score(y_test, y_pred_svc, average=None)
print("SVC f1:", f1 )


# The F1 score is best with class_0, followed by class_1 and finally class_2.

# In[33]:


conf_mat = metrics.confusion_matrix(y_test, y_pred_svc)
print("SVC conf_mat: %f")
print(conf_mat)


# Based on all the scores a classification report can be generated for the SVM Classification Model.

# In[34]:


print("SVM Classification Report:\n", metrics.classification_report(y_test, y_pred_svc, digits=4))


# # K-Nearest Neighbors

# I will first create a KNN Classifier object and define a parameter grid for parameter tuning.
# For the param_grid argument, I created a dictionary and chose 3 parameters used in the Random Forest Classifer.
# n_neighbors used for the number of neighbors to use - default (5)
# weights function is used for prediction - default (uniform)
# p is the power parameter for the Minkowski Metric - default (2)

# In[35]:


from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()

param_grid3 = {
    'n_neighbors': [3, 5, 7, 9],
    'p': [1, 2],
    'weights': ['uniform', 'distance']
}


# In[36]:


grid_search3 = GridSearchCV(knn_clf, param_grid=param_grid3, cv=5)
grid_search3.fit(X_train, y_train)


# I created a GridSearchCV object and included the estimator, the param_grid and set the K-fold Cross Validation to 5 which is the default for determining the cross-validation splitting strategy. Then I Fit the GridSearchCV object to the training data.

# In[37]:


print("Best Parameters: ", grid_search3.best_estimator_)
print("Best Parameters:", grid_search3.best_params_)


# As shown the best parameters were 5 which is the default for n_neighbors, 1 for p which uses manhattan_distance and uniform which is the default value for weights. Then the best estimator object is fit to the training data, and the classifier is evaluated.

# In[38]:


best_knn = grid_search3.best_estimator_
best_knn.fit(X_train, y_train)
y_train_pred_knn = best_knn.predict(X_train)
y_pred_knn = best_knn.predict(X_test)

acc_train = metrics.accuracy_score(y_train, y_train_pred_knn)
print("kNN acc_train:", acc_train )
acc = metrics.accuracy_score(y_test, y_pred_knn)
print("kNN acc:", acc )


# The accuracy score in % for kNN is 94.4

# In[39]:


recall_train = metrics.recall_score(y_train, y_train_pred_knn, average=None)
print("kNN recall_train:", recall_train)
recall = metrics.recall_score(y_test, y_pred_knn, average=None)
print("kNN recall:", recall)


# The recall score is best in class_0 and class class_2, but low in class_1.

# In[40]:


prec_train = metrics.precision_score(y_train, y_train_pred_knn, average=None)
print("kNN precision_train:", prec_train )
prec = metrics.precision_score(y_test, y_pred_knn, average=None)
print("kNN precision:", prec )


# The Precision score is best in class_1 followed by class_0 and then class_2.

# In[41]:


f1_train = metrics.f1_score(y_train, y_train_pred_knn, average=None)
print("kNN f1 train:", f1_train )
f1 = metrics.f1_score(y_test, y_pred_knn, average=None)
print("kNN f1:", f1 )


# The F1 score is best in class_0, followed by class_2 and finally class_1.

# In[42]:


conf_mat = metrics.confusion_matrix(y_test, y_pred_knn)
print("kNN Confusion Matrix:")
print(conf_mat)


# Based on all the scores a classification report can be generated for the Decision Tree Classification Model.

# In[43]:


print("K-Nearest Neighbors Classification Report:\n", metrics.classification_report(y_test, y_pred_knn, digits=4))


# # Accuracy, Precision, Recall, F1 Scores and Confusion Matrix

# Accuracy is the ratio of correct predictions out of all predictions made by an algorithm. It can be calculated by dividing precision by recall or as 1 minus false negative rate (FNR) divided by false positive rate (FPR).
# 
# Accuracy = ((TP + TN)) / ((TP + FP + TN + FN))
# 
# The Precision is the ratio of true positives over the sum of false positives and true negatives. It is also known as positive predictive value. Precision is a useful metric and shows that out of those predicted as positive, how accurate the prediction was.
# 
# Precision = ((TP)/(TP + FP)) = ((TP)/(Total Predicted Positive))
# 
# Recall is the ratio of correctly predicted outcomes to all predictions. It is also known as sensitivity or specificity. Recall is just the proportion of positives our model are able to catch through labelling them as positives. When the cost of False Negative is greater than that of False Positive, we should select our best model using Recall.
# 
# Recall = ((TP)/(TP + FN)) = ((TP)/(Total Actual Positive))
# 
#  F1 score is a weighted average of precision and recall. As we know in precision and in recall there is false positive and false negative so it also consider both of them. F1 score is usually more useful than accuracy, especially if you have an uneven class distribution
# 
# F1 Score = 2*(Recall * Precision) / (Recall + Precision)
# 
# The confusion matrix is a table that summarizes how successful the classification model is at predicting examples belonging to various classes. One axis of the confusion matrix is the label that the model predicted, and the other axis is the actual label.  We can use confusion matrix when we compare different model by looking how well it predicted a true positive(TP) and true negative(TN). If one model predicted a TP and TN very well than other model then we choose this model as our base model.

# # Conclusion

# The highest accuracy was achieved by the Random Forest Classifier - 100% accuracy
# The 2nd highest accuracy was achieved by Support Vector Machines -  97.2% accuracy
# The 3rd highest accuracy was achieved by both Decision Tree & K-Nearest Neighbors classifiers - 94.4% accuracy
# 
# I believe that tuning the parameters affected the evaluation and performance of the classification models, and with GridSearchCV the best parameters were chosen for each algorithm for the best results.
# 
# Additionally from observing the Precision, Recall and F1 Scores of each classification models, it is seen that Random Forest Classifier performs best once again in all classes. This is followed by Support Vector Machines, then Decision Tree Classiifer & K-Nearest Neighbors, with a slightly better average score in Decision Tree Classifier.
# 
# In conclusion, it is safe to say that Random Forest Classifer obtains absolute accuracy making it the best classfication model, followed by SVM, K-Nearest Neighbors and finally Decision Tree Classifers with pretty decent accuracy scores.
