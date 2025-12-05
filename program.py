###Decision trees
"""
Desicion trees is a classification algorithm, and as the name suggests, uses a tree based model to classify instances. Each node (except leaf level nodes) will filter out instances based on a condition. At leaf nodes, the instances will be assigned with a class label.
 mporting required libraries
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading data from input csv file
credit_data = pd.read_csv("Decision Trees\datasets\credit_risk.csv")
# Viewing sample data
print(credit_data.head())
""" Feature Engineering
Let us now look at the data to get insights on it, which will help us build a good model.
"""
print(credit_data.info())
# Selecting the predictor attributes
X = credit_data.columns.drop("class")

# Selecting  the target
y = credit_data['class']
#### Encoding the categorical variables
# Encoding all the predictor variables using the get_dummies method() to convert the categorical values to numerical values.
credit_data_encoded = pd.get_dummies(credit_data[X])

# Uncomment the next line to see the list of all column names
#credit_data_encoded.columns
print("Total number of predictors after encoding = ", len(credit_data_encoded.columns))
#### Splitting the data into train and test set in a ratio of 85:15
from sklearn.model_selection import train_test_split

#splitting data into train and test datasets
X_train,X_test,y_train,y_test = train_test_split(credit_data_encoded, y,test_size=0.15,random_state=100) 

# Printing the shape of the resulting datasets
print("Shape of X_train and y_train are:", X_train.shape, "and", y_train.shape, " respectively")
print("Shape of X_test and y_test are:", X_test.shape, "and", y_test.shape, " respectively")
# Importing required class 
from sklearn.tree import DecisionTreeClassifier

# Creating an object of the DecisionTreeClassifier model
model = DecisionTreeClassifier(random_state = 1)

# Training model on the training data
print(model.fit(X_train,y_train))
#predicting targets based on the model built
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
#### Visualize the tree
from sklearn import tree as sktree

# Visualize the tree using matplotlib (no external Graphviz executable required)
plt.figure(figsize=(20, 10))
sktree.plot_tree(
    model,
    feature_names=credit_data_encoded.columns,
    class_names=[str(c) for c in model.classes_],
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=300)
plt.show()
print("Decision tree plot saved to decision_tree.png")

#### Evaluate model performance on train and test data
# Getting the accuracy on train data
train_accuracy = model.score(X_train,y_train)
print("Accuracy of the model on train data = ",train_accuracy)

# Getting the accuracy on test data
test_accuracy = model.score(X_test,y_test)
print("Accuracy of the model on test data = ",test_accuracy)
#### Tuning the hyper-parameters
"""
Here you can observe that the tree looks like an overfit model. It has a 100% accuracy in train and just 67% in test. 
To avoid this problem, we need to tune certain parameters of the tree algorithm called hyper parameters
min_samples_split - number of data instances required in a node to proceed with further splitting of node.

min_impurity_decrease - at every level of the decision tree, the data instances gets concentrated towards one of the
class labels. This corresponds to the decrease in impurity of dataset in the node. so when min_impurity_decrease is specified, 
nodes will be further split only when impurity decreases by the specified value.
"""
# Model 1:
# Min number of samples required in a set to split = 10
# Min reduction in impurity required for split to be included in the tree = 0.005

model1 = DecisionTreeClassifier(min_samples_split=10,min_impurity_decrease=0.005)

# Fitting the model to the training data
model1.fit(X_train,y_train)

# Measuring the accuracy of the model
print("Model1 train accuracy = ", model1.score(X_train,y_train))
print("Model1 test accuracy = ", model1.score(X_test,y_test))
# Model 2:
# Min number of samples required in a set to split = 20
# Min reduction in impurity required for split to be included in the tree = 0.1

model2 = DecisionTreeClassifier(min_samples_split=20,min_impurity_decrease=0.1)

# Fitting the model to the training data
model2.fit(X_train,y_train)

# Measuring the accuracy of the model
print("Model2 train accuracy = ", model2.score(X_train,y_train))
print("Model2 test accuracy = ", model2.score(X_test,y_test))
""" Confusion Matrix
Confusion matrix helps to assess how good the model works on individual classes in the outcome
"""
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#confusion matrix compares the actual target values and the predicted target values
train_conf_matrix = confusion_matrix(y_train,train_predictions)
test_conf_matrix = confusion_matrix(y_test,test_predictions)
print(pd.DataFrame(train_conf_matrix,columns=model.classes_,index=model.classes_))
print(pd.DataFrame(test_conf_matrix,columns=model.classes_,index=model.classes_))
#train accuracy calculated from confusion matrix
train_correct_predictions = train_conf_matrix[0][0]+train_conf_matrix[1][1]
train_total_predictions = train_conf_matrix.sum()
train_accuracy = train_correct_predictions/train_total_predictions
print(train_accuracy)
#test accuracy calculated from confusion matrix
test_correct_predictions = test_conf_matrix[0][0]+test_conf_matrix[1][1]
total_predictions = test_conf_matrix.sum()
test_accuracy = test_correct_predictions/total_predictions
print(test_accuracy)
###Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_train,train_predictions))
print(classification_report(y_test,test_predictions))